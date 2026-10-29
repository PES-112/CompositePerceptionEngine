import ARKit
import CoreImage
import Foundation
import os

/// The whole product on the phone, one processed frame every 0.2 s (5 Hz, as the replay):
///
///   ARKit (camera image + LiDAR scene depth + camera pose, held portrait)
///     -> Stage 1: YOLO26n v3 -> IoU tracker -> LiDAR depth per box, closing speed, bearing
///     -> segmentation (every 2nd frame, 2.5 Hz): walker surface, edge ahead, per-object context
///     -> threat engine: Kalman fusion with ego-motion removed, safety envelope, GRU arrival,
///        fused priority, announcement manager (override / edge / threat / surface lanes)
///     -> phrase -> on-device speech, panned toward the object
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
    private let speech = Speech()

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
        running = true
        status = conf.frameSemantics.contains(.smoothedSceneDepth) || conf.frameSemantics.contains(.sceneDepth)
            ? "Running (LiDAR depth)" : "Running — no LiDAR on this device: distances unavailable"
        speech.say("C P E started.", bearingDeg: 0, interrupt: true)
    }

    func stop() {
        session.pause()
        speech.stop()
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
        var input = Self.input(frame, t: t - (t0 ?? t))
        if t0 == nil { t0 = t }
        if let p = input.pose { lastPose = p } else { input = input.with(pose: lastPose) }   // keep the world frame
        queue.async { [weak self] in
            self?.process(input)
            self?.arQueue.async { self?.busy = false }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { self.status = "ARKit error: \(error.localizedDescription)" }
    }

    // MARK: one processed frame

    private func process(_ f: FrameInput) {
        guard let stage1, let engine, let narrator, let edges, let surfaces, let stairs else { return }
        let started = DispatchTime.now().uptimeNanoseconds
        do {
            let rows = try stage1.process(f)
            var edgeAlert: EdgeAlert?
            if let seg = segmenter, k % 2 == 0, let map = try seg.classMap(f) {
                lastScene = seg.sceneFrame(map, f.depth, rows: rows, imageWidth: Double(f.size.width))
                edgeAlert = edges.step(t: f.t, edge: lastScene.edge, label: seg.label)
            }
            let surfaceID = surfaces.step(k % 2 == 0 ? lastScene.walkerSurface : nil)
            var surfaceName = surfaceID.flatMap { segmenter?.label($0) }
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
            k += 1
            var phrase: String?
            if let a = frame.announcement {
                let track = frame.tracks.first { $0.trackID == a.trackID }
                let (text, bearing) = narrator.phrase(a, track: track)
                phrase = text
                let interrupt = a.kind == .override
                DispatchQueue.main.async { self.speech.say(text, bearingDeg: bearing, interrupt: interrupt) }
                log.info("t=\(f.t, format: .fixed(precision: 1)) \(a.kind.rawValue, privacy: .public): \(text, privacy: .public)")
            }
            let ranked = Set(frame.ranked.map(\.track.trackID)), over = Set(frame.overrides.map(\.track.trackID))
            let ov = rows.map { r in
                Overlay(box: r.box, label: "\(r.className) \(r.distanceM.map { String(format: "%.1fm", $0) } ?? "?")",
                        color: over.contains(r.trackID) ? 2 : ranked.contains(r.trackID) ? 1 : 0)
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
            DispatchQueue.main.async {
                self.overlays = ov
                self.imageSize = f.size
                self.latencyMs = ms
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
        return FrameInput(t: t, image: upright, size: CGSize(width: ext.width, height: ext.height),
                          depth: depthData.map(Self.depth), focalPx: fy, principalX: sensorH - 1 - cy,
                          pose: Self.pose(frame.camera.transform),
                          camera: SIMD3(Double(frame.camera.transform.columns.3.x), -Double(frame.camera.transform.columns.3.z),
                                        Double(frame.camera.transform.columns.3.y)))
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
