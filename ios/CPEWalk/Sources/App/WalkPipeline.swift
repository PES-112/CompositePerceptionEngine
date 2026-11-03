import ARKit
import CoreImage
import Foundation
import os

/// The whole product on the phone, one processed frame every 0.2 s (5 Hz, as the replay):
///
///   ARKit (camera image + LiDAR scene depth + camera pose, held portrait)
///     -> Stage 1: YOLO26n v4 IDD + v3 legacy -> IoU tracker -> LiDAR depth per box, closing speed, bearing
///     -> street rules (phone only): vehicles approaching while crossing, posts in the path,
///        LiDAR "Obstacle ahead"
///     -> threat engine: Kalman fusion with ego-motion removed, safety envelope, GRU arrival,
///        fused priority, announcement manager (override / edge / threat / surface lanes)
///     -> phrase -> on-device speech, panned toward the object
///     -> segmentation (every 2nd frame, 2.5 Hz), after the alert is spoken: walker surface, edge
///        ahead, per-object context, used from the next frame on. It is the slowest stage (~87 ms),
///        so it stays off the alert path; the replay uses each frame's own context (0.2 s sooner).
///
/// ARKit's world tracking stands in for SANPO's fixed_camera_poses.csv, and the iPhone's LiDAR
/// for SANPO's stereo depth.
final class WalkPipeline: NSObject, ObservableObject, ARSessionDelegate {
    struct Overlay { let box: CGRect; let label: String; let color: Int }   // 0 grey, 1 ranked, 2 override

    @Published var running = false
    @Published var status = "Ready"
    @Published var lastPhrase = ""
    @Published var surface = "—"
    @Published var latencyMs = 0.0
    @Published var perfFile = ""                 // CSV of this walk's per-frame timing (PerfLog)
    @Published var overlays: [Overlay] = []
    @Published var imageSize = CGSize(width: 1440, height: 1920)
    @Published var surfaceReminders = true

    let session = ARSession()
    private let queue = DispatchQueue(label: "cpe.pipeline", qos: .userInitiated)
    private let arQueue = DispatchQueue(label: "cpe.arkit")
    private var lastPose: EgoPose?
    private let log = Logger(subsystem: "edu.pes.cpe.walk", category: "pipeline")
    private var busy = false
    private var lastProcessed = -Double.infinity
    private var t0: Double?
    private var k = 0

    private var cfg: CPEConfig?
    private var stage1: Stage1?
    private var segmenter: Segmenter?
    private var engine: ThreatEngine?
    private var narrator: Narrator?
    private var edges: EdgeAlerter?
    private var surfaces: SurfaceTracker?
    private var stairs: StairMotion?
    private var lastScene = SceneFrame(walkerSurface: nil, edge: nil)
    private var sceneEdge: EdgeAlert?            // from the last segmentation, for the next frame
    private var freshScene = false
    private var crossing = CrossingWatch()
    private var posts = PostWatch()
    private var obstacles = PathObstacleAlerter()
    private let speech = Speech()
    private var perf: PerfLog?

    /// One processed frame, for the SANPO replay (Tests/SanpoReplayTests.swift): what was said and the
    /// stage times. `speakAloud = false` keeps the replay silent.
    struct FrameResult { let t: Double; let kind: String?; let className: String?; let phrase: String?
                         let bearingDeg: Double?; let surface: String?; let timing: PerfLog.Row; let overlays: [Overlay] }
    var onFrame: ((FrameResult) -> Void)?
    var speakAloud = true

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = arQueue
    }

    func load() throws {
        let c = try CPEConfig.load()
        cfg = c
        stage1 = Stage1(config: c, detectors: try Detector.load(config: c))
        segmenter = try? Segmenter(config: c)        // optional: the engine runs without context
        engine = ThreatEngine(config: c, predictor: try GRUArrivalPredictor(config: c))
        narrator = Narrator(n: c.narration)
        edges = EdgeAlerter(c.segmentation)
        surfaces = SurfaceTracker(confirm: c.segmentation.surface_confirm_frames)
        stairs = StairMotion(c.stairs)
        crossing = CrossingWatch(); posts = PostWatch(); obstacles = PathObstacleAlerter()
        lastScene = SceneFrame(walkerSurface: nil, edge: nil); sceneEdge = nil; freshScene = false
    }

    func start() {
        guard ARWorldTrackingConfiguration.isSupported else { status = "ARKit world tracking not supported"; return }
        do { try load() } catch { status = "Load failed: \(error.localizedDescription)"; return }
        let conf = ARWorldTrackingConfiguration()
        lastPose = nil
        conf.worldAlignment = .gravity
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            conf.frameSemantics.insert(.smoothedSceneDepth)
        } else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            conf.frameSemantics.insert(.sceneDepth)
        }
        session.run(conf, options: [.resetTracking, .removeExistingAnchors])
        t0 = nil; k = 0; lastProcessed = -.infinity
        perf = try? PerfLog(detectors: stage1?.detectors.count ?? 0, segmenter: segmenter != nil)
        perfFile = perf?.url.lastPathComponent ?? ""
        running = true
        status = conf.frameSemantics.contains(.smoothedSceneDepth) || conf.frameSemantics.contains(.sceneDepth)
            ? "Running (LiDAR depth)" : "Running — no LiDAR on this device: distances unavailable"
        speech.say("C P E started.", bearingDeg: 0, interrupt: true)
    }

    func stop() {
        session.pause()
        speech.stop()
        let p = perf
        perf = nil
        queue.async { p?.close() }          // after the frame in flight
        running = false
        status = "Stopped"
        overlays = []
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let t = frame.timestamp
        guard running, !busy, t - lastProcessed >= 0.2 - 0.008 else { return }
        if case .notAvailable = frame.camera.trackingState { return }
        busy = true
        lastProcessed = t
        let s0 = DispatchTime.now().uptimeNanoseconds
        var input = Self.input(frame, t: t - (t0 ?? t))
        if t0 == nil { t0 = t }
        if let p = input.pose { lastPose = p } else { input = input.with(pose: lastPose) }   // keep the world frame
        let inputMs = msSince(s0)
        queue.async { [weak self] in
            self?.process(input, inputMs: inputMs)
            self?.arQueue.async { self?.busy = false }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { self.status = "ARKit error: \(error.localizedDescription)" }
    }

    // MARK: one processed frame

    func process(_ f: FrameInput, inputMs: Double) {
        guard let stage1, let engine, let narrator, let edges, let surfaces, let stairs else { return }
        let started = DispatchTime.now().uptimeNanoseconds
        var timing = PerfLog.Row(inputMs: inputMs)
        do {
            let rows = try stage1.process(f)
            let dets = stage1.detectMs
            timing.det1Ms = dets.first ?? 0
            timing.det2Ms = dets.dropFirst().reduce(0, +)
            timing.trackMs = msSince(started) - dets.reduce(0, +)
            timing.tracks = rows.count
            var edgeAlert = sceneEdge
            sceneEdge = nil
            let surfaceID = surfaces.step(freshScene ? lastScene.walkerSurface : nil)
            freshScene = false
            var surfaceName = surfaceID.flatMap { segmenter?.label($0) }

            // Street rules (StreetRules.swift); all run every frame to keep their history.
            let rulesStart = DispatchTime.now().uptimeNanoseconds
            let atRoadEdge = lastScene.edge.map { $0.kind == Ground.road && $0.distanceM <= 3 } ?? false
            let onRoad = surfaceName == "road" || surfaceName == "crosswalk" || atRoadEdge
            let crossingAlert = crossing.step(t: f.t, rows: rows, pose: f.pose, focalPx: f.focalPx, onRoad: onRoad)
            let postAlert = posts.step(t: f.t, rows: rows)
            var obstacleM: Double?
            if let d = f.depth, let g = f.depthGeometry, let pose = f.pose {
                let sx = Double(d.width) / Double(f.size.width), sy = Double(d.height) / Double(f.size.height)
                let named = rows.filter { !StreetRules.groundClasses.contains($0.className) }   // already named by a lane
                    .map { CGRect(x: $0.box.minX * sx, y: $0.box.minY * sy, width: $0.box.width * sx, height: $0.box.height * sy) }
                let scan = PathObstacleScan.scan(d, g, pose: pose, exclude: named)
                obstacleM = PathObstacleScan.nearest(scan.points)
                timing.cameraHeightM = scan.cameraHeightM
                timing.obstacleM = obstacleM
            }
            let obstacleAlert = obstacles.step(t: f.t, distanceM: postAlert == nil ? obstacleM : nil)
            edgeAlert = crossingAlert ?? edgeAlert ?? postAlert ?? obstacleAlert
            timing.rulesMs = msSince(rulesStart)

            let engineStart = DispatchTime.now().uptimeNanoseconds
            // On a flight, the walker's height says more than the segmenter (which sees 2-3 m ahead):
            // stair events take the edge lane, and edge / surface phrases from the class map wait.
            if let c = f.camera, let kind = stairs.step(t: f.t, x: c.x, y: c.y, h: c.z) {
                edgeAlert = EdgeAlert(t: f.t, kind: kind, distanceM: 0)
            } else if stairs.onStairs {
                edgeAlert = nil
            }
            if stairs.onStairs { surfaceName = "stairs" }
            let frame = try engine.step(k: k, t: f.t, rows: rows, pose: f.pose, contexts: lastScene.contexts,
                                        edge: edgeAlert, surface: surfaceReminders && !stairs.onStairs ? surfaceName : nil)
            var phrase: String?
            var spokenBearing: Double?
            if let a = frame.announcement {
                let track = frame.tracks.first { $0.trackID == a.trackID }
                let (text, bearing) = narrator.phrase(a, track: track)
                phrase = text
                spokenBearing = bearing
                let interrupt = a.kind == .override
                if speakAloud { DispatchQueue.main.async { self.speech.say(text, bearingDeg: bearing, interrupt: interrupt) } }
                log.info("t=\(f.t, format: .fixed(precision: 1)) \(a.kind.rawValue, privacy: .public): \(text, privacy: .public)")
            }
            timing.engineMs = msSince(engineStart)
            timing.alertMs = inputMs + msSince(started)      // camera frame -> alert handed to speech

            // Segmentation last, off the alert path; its scene is used from the next frame.
            let segStart = DispatchTime.now().uptimeNanoseconds
            if let seg = segmenter, k % 2 == 0, let map = try seg.classMap(f) {
                lastScene = seg.sceneFrame(map, f.depth, rows: rows, imageWidth: Double(f.size.width))
                sceneEdge = edges.step(t: f.t, edge: lastScene.edge, label: seg.label)
                freshScene = true
            }
            timing.segMs = msSince(segStart)
            k += 1
            let ranked = Set(frame.ranked.map(\.track.trackID)), over = Set(frame.overrides.map(\.track.trackID))
            let ov = rows.map { r in
                Overlay(box: r.box, label: "\(r.className) \(r.distanceM.map { String(format: "%.1fm", $0) } ?? "?")",
                        color: over.contains(r.trackID) ? 2 : ranked.contains(r.trackID) ? 1 : 0)
            }
            onFrame?(FrameResult(t: f.t, kind: frame.announcement?.kind.rawValue, className: frame.announcement?.className,
                                 phrase: phrase, bearingDeg: spokenBearing, surface: surfaceName, timing: timing, overlays: ov))
            let ms = msSince(started)
            timing.totalMs = inputMs + ms
            perf?.add(t: f.t, timing)
            DispatchQueue.main.async {
                self.overlays = ov
                self.imageSize = f.size
                self.latencyMs = timing.totalMs
                if let phrase { self.lastPhrase = phrase }
                self.surface = surfaceName ?? "—"
            }
        } catch {
            DispatchQueue.main.async { self.status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: ARFrame -> FrameInput (portrait)

    /// ARKit delivers the image and depth in the sensor's landscape-right orientation; the phone is
    /// held portrait, so both are rotated 90° clockwise (.right). A sensor pixel (x, y) of a W x H
    /// image lands at portrait (H - 1 - y, x).
    static func input(_ frame: ARFrame, t: Double) -> FrameInput {
        let img = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
        let ext = img.extent
        let upright = img.transformed(by: CGAffineTransform(translationX: -ext.minX, y: -ext.minY))
        let sensorH = Double(CVPixelBufferGetHeight(frame.capturedImage))
        let K = frame.camera.intrinsics
        let fy = Double(K.columns.1.y), cy = Double(K.columns.2.y)
        let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth
        var geometry: DepthGeometry?
        if let dm = depthData?.depthMap {
            let s = Double(CVPixelBufferGetWidth(dm)) / Double(CVPixelBufferGetWidth(frame.capturedImage))
            let m = frame.camera.transform
            let col = { (c: SIMD4<Float>) in SIMD3(Double(c.x), Double(c.y), Double(c.z)) }
            geometry = DepthGeometry(fx: Double(K.columns.0.x) * s, fy: Double(K.columns.1.y) * s,
                                     cx: Double(K.columns.2.x) * s, cy: Double(K.columns.2.y) * s,
                                     sensorHeight: CVPixelBufferGetHeight(dm),
                                     rotation: simd_double3x3(columns: (col(m.columns.0), col(m.columns.1), col(m.columns.2))),
                                     position: col(m.columns.3))
        }
        return FrameInput(t: t, image: upright, size: CGSize(width: ext.width, height: ext.height),
                          depth: depthData.map(Self.depth), focalPx: fy, principalX: sensorH - 1 - cy,
                          pose: Self.pose(frame.camera.transform),
                          camera: SIMD3(Double(frame.camera.transform.columns.3.x), -Double(frame.camera.transform.columns.3.z),
                                        Double(frame.camera.transform.columns.3.y)),
                          depthGeometry: geometry)
    }

    static func depth(_ d: ARDepthData) -> DepthMap {
        let map = d.depthMap, conf = d.confidenceMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        if let conf { CVPixelBufferLockBaseAddress(conf, .readOnly) }
        defer {
            CVPixelBufferUnlockBaseAddress(map, .readOnly)
            if let conf { CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
        }
        let W = CVPixelBufferGetWidth(map), H = CVPixelBufferGetHeight(map)
        let rowD = CVPixelBufferGetBytesPerRow(map) / MemoryLayout<Float32>.stride
        let pd = CVPixelBufferGetBaseAddress(map)!.assumingMemoryBound(to: Float32.self)
        let pc = conf.map { CVPixelBufferGetBaseAddress($0)!.assumingMemoryBound(to: UInt8.self) }
        let rowC = conf.map(CVPixelBufferGetBytesPerRow) ?? 0
        // portrait: width H, height W; portrait (u, v) <- sensor (x = v, y = H - 1 - u)
        var out = [Float](repeating: 0, count: W * H)
        for v in 0..<W { for u in 0..<H {
            let x = v, y = H - 1 - u
            let low = pc.map { $0[y * rowC + x] < UInt8(ARConfidenceLevel.medium.rawValue) } ?? false
            out[v * H + u] = low ? 0 : pd[y * rowD + x]
        } }
        return DepthMap(width: H, height: W, values: out)
    }

    /// ARKit world (y up) -> top view (X = x, Y = -z); camera looks along -z.
    static func pose(_ m: simd_float4x4) -> EgoPose? {
        let f3 = -SIMD3<Double>(Double(m.columns.2.x), Double(m.columns.2.y), Double(m.columns.2.z))
        var fwd = SIMD2(f3.x, -f3.z)
        let n = simd_length(fwd)
        guard n > 0.2 else { return nil }                      // pointing at the floor or the sky
        fwd /= n
        return EgoPose(position: SIMD2(Double(m.columns.3.x), -Double(m.columns.3.z)),
                       right: SIMD2(fwd.y, -fwd.x), forward: fwd)
    }
}
