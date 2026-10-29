import CoreGraphics
import CoreImage
import CoreML
import Foundation

// Stage 1 on the phone (src/perception_stack): detector -> tracker -> depth per box -> closing speed
// and pinhole bearing, giving the same row fields the replay and the GRU were built on.

struct Detection { var box: CGRect; var cls: Int; var conf: Double }

/// An upright camera frame plus its depth, both in portrait image coordinates.
struct FrameInput {
    let t: Double
    let image: CIImage                 // upright (portrait), width W x height H
    let size: CGSize
    let depth: DepthMap?               // metres, upright, any resolution
    let focalPx: Double                // horizontal focal length in `image` pixels
    let principalX: Double
    let pose: EgoPose?
    var camera: SIMD3<Double>? = nil    // camera position: top view (X, Y) and height, for stairs.StairMotion

    func with(pose p: EgoPose?) -> FrameInput {
        FrameInput(t: t, image: image, size: size, depth: depth, focalPx: focalPx, principalX: principalX, pose: p,
                   camera: camera)
    }
}

struct DepthMap {
    let width: Int
    let height: Int
    var values: [Float]                // row-major, metres; 0 / non-finite = no reading
    func at(_ x: Int, _ y: Int) -> Float { values[y * width + x] }
}

// MARK: - Detector (YOLO26n v3, raw head: [1, 4 + C, 8400], cx cy w h in 640-px letterbox coords)

final class Detector {
    private let model: MLModel
    private let inputName: String
    private let size: Int
    private let conf: Double
    private let iou: Double
    private let maxDet: Int
    private let classes: Set<Int>?      // keep only these class ids (nil = all)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var buffer: CVPixelBuffer?

    /// The detectors this app bundle runs. With CPEDetectorLegacy present and
    /// detector.legacy_classes set, the IDD-tuned CPEDetector keeps the other classes and the
    /// original detector answers for legacy_classes (methodology §4.26); otherwise one detector.
    static func load(config: CPEConfig, bundle: Bundle = .main) throws -> [Detector] {
        let legacy = Set(config.detector.legacy_classes.compactMap { config.classes.firstIndex(of: $0) })
        guard !legacy.isEmpty, bundle.url(forResource: "CPEDetectorLegacy", withExtension: "mlmodelc") != nil else {
            return [try Detector(config: config, bundle: bundle)]
        }
        let others = Set(config.classes.indices).subtracting(legacy)
        return [try Detector(config: config, bundle: bundle, classes: others),
                try Detector(config: config, bundle: bundle, name: "CPEDetectorLegacy", classes: legacy)]
    }

    init(config: CPEConfig, bundle: Bundle = .main, name: String = "CPEDetector", classes: Set<Int>? = nil) throws {
        guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
            throw CPEError.missing("\(name).mlpackage") }
        let mc = MLModelConfiguration()
        mc.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: mc)
        inputName = model.modelDescription.inputDescriptionsByName.first { $0.value.type == .image }?.key ?? "image"
        size = config.detector.imgsz
        conf = config.detector.conf
        iou = config.detector.nms_iou
        maxDet = config.detector.max_det
        self.classes = classes
    }

    /// Letterbox `image` into a size x size BGRA buffer (grey 114 padding, as Ultralytics). Returns the
    /// buffer, the scale and the padding, to map boxes back.
    static func letterbox(_ image: CIImage, size: Int, context: CIContext, into buf: inout CVPixelBuffer?)
        -> (CVPixelBuffer, Double, CGPoint)? {
        if buf == nil {
            let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
            CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, attrs, &buf)
        }
        guard let out = buf else { return nil }
        let ext = image.extent
        let r = min(Double(size) / ext.width, Double(size) / ext.height)
        let w = ext.width * r, h = ext.height * r
        let pad = CGPoint(x: (Double(size) - w) / 2, y: (Double(size) - h) / 2)
        let grey = CIImage(color: CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255))
            .cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
        let placed = image.transformed(by: CGAffineTransform(translationX: -ext.minX, y: -ext.minY)
            .concatenating(CGAffineTransform(scaleX: r, y: r))
            .concatenating(CGAffineTransform(translationX: pad.x, y: pad.y)))
        context.render(placed.composited(over: grey), to: out)
        return (out, r, pad)
    }

    func detect(_ frame: FrameInput) throws -> [Detection] {
        guard let (pb, r, pad) = Self.letterbox(frame.image, size: size, context: ciContext, into: &buffer) else { return [] }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pb)]))
        guard let name = out.featureNames.first, let a = out.featureValue(for: name)?.multiArrayValue else { return [] }
        let rows = a.shape[1].intValue, n = a.shape[2].intValue, nc = rows - 4
        let s0 = a.strides[1].intValue, s1 = a.strides[2].intValue
        var cand: [Detection] = []
        let ptr = a.dataType == .float16 ? nil : a.dataPointer.assumingMemoryBound(to: Float.self)
        func v(_ row: Int, _ i: Int) -> Double {
            if let ptr { return Double(ptr[row * s0 + i * s1]) }
            return a[[0, NSNumber(value: row), NSNumber(value: i)] as [NSNumber]].doubleValue
        }
        for i in 0..<n {
            var best = 0, bestScore = 0.0
            for c in 0..<nc { let s = v(4 + c, i); if s > bestScore { bestScore = s; best = c } }
            guard bestScore >= conf, classes?.contains(best) ?? true else { continue }   // the other model's class
            let cx = v(0, i), cy = v(1, i), w = v(2, i), h = v(3, i)
            // letterbox (top-left origin) -> image pixels (top-left origin)
            let x1 = (cx - w / 2 - pad.x) / r, y1 = (cy - h / 2 - pad.y) / r
            let box = CGRect(x: x1, y: y1, width: w / r, height: h / r)
                .intersection(CGRect(origin: .zero, size: frame.size))
            if !box.isNull && box.width > 1 && box.height > 1 { cand.append(Detection(box: box, cls: best, conf: bestScore)) }
        }
        return Self.nms(cand, iou: iou, maxDet: maxDet)
    }

    /// Per-class greedy NMS (Ultralytics' default, agnostic = False).
    static func nms(_ d: [Detection], iou: Double, maxDet: Int) -> [Detection] {
        var keep: [Detection] = []
        for det in d.sorted(by: { $0.conf > $1.conf }) {
            if keep.contains(where: { $0.cls == det.cls && IoU($0.box, det.box) > iou }) { continue }
            keep.append(det)
            if keep.count >= maxDet { break }
        }
        return keep
    }
}

func IoU(_ a: CGRect, _ b: CGRect) -> Double {
    let i = a.intersection(b)
    if i.isNull { return 0 }
    let inter = Double(i.width * i.height)
    return inter / (Double(a.width * a.height + b.width * b.height) - inter)
}

// MARK: - Tracker: ByteTrack's association, simplified

/// Persistent IDs by IoU, as Ultralytics' ByteTrack does for detections above its high threshold
/// (Stage 1 runs the detector at conf 0.30, so ByteTrack's low-score second pass sees almost
/// nothing on the replay either). Boxes are predicted forward at constant velocity between frames;
/// a lost track is kept for `buffer` frames.
final class BoxTracker {
    private struct Track { var id: Int; var box: CGRect; var vel: CGVector; var cls: Int; var lost: Int }
    private var tracks: [Track] = []
    private var nextID = 1
    private let matchIoU: Double
    private let buffer: Int

    init(matchIoU: Double = 0.2, buffer: Int = 30) { self.matchIoU = matchIoU; self.buffer = buffer }

    func update(_ dets: [Detection]) -> [(id: Int, det: Detection)] {
        let predicted = tracks.map { $0.box.offsetBy(dx: $0.vel.dx, dy: $0.vel.dy) }
        var pairs: [(Double, Int, Int)] = []
        for (ti, pb) in predicted.enumerated() {
            for (di, d) in dets.enumerated() {
                let o = IoU(pb, d.box)
                if o >= matchIoU { pairs.append((o, ti, di)) }
            }
        }
        var usedT = Set<Int>(), usedD = Set<Int>(), out: [(Int, Detection)] = []
        for (_, ti, di) in pairs.sorted(by: { $0.0 > $1.0 }) where !usedT.contains(ti) && !usedD.contains(di) {
            usedT.insert(ti); usedD.insert(di)
            let old = tracks[ti].box, nb = dets[di].box
            tracks[ti].vel = CGVector(dx: 0.5 * (nb.midX - old.midX) + 0.5 * tracks[ti].vel.dx,
                                      dy: 0.5 * (nb.midY - old.midY) + 0.5 * tracks[ti].vel.dy)
            tracks[ti].box = nb
            tracks[ti].cls = dets[di].cls
            tracks[ti].lost = 0
            out.append((tracks[ti].id, dets[di]))
        }
        for ti in tracks.indices where !usedT.contains(ti) { tracks[ti].lost += 1 }
        tracks.removeAll { $0.lost > buffer }
        for (di, d) in dets.enumerated() where !usedD.contains(di) {
            tracks.append(Track(id: nextID, box: d.box, vel: .zero, cls: d.cls, lost: 0))
            out.append((nextID, d))
            nextID += 1
        }
        return out
    }
}

// MARK: - Stage-1 rows: depth per box, closing speed, bearing

final class Stage1 {
    private let cfg: CPEConfig
    let detectors: [Detector]
    private let tracker = BoxTracker()
    private var depthHistory: [String: [(Double, Double)]] = [:]     // (t, metres)

    init(config: CPEConfig, detectors: [Detector]) { cfg = config; self.detectors = detectors }

    func process(_ frame: FrameInput) throws -> [PerceptionRow] {
        let W = Double(frame.size.width), H = Double(frame.size.height)
        let dets = try detectors.flatMap { try $0.detect(frame) }.filter { d in        // yolo_tracker.is_valid_detection
            let bw = Double(d.box.width), bh = Double(d.box.height)
            return bw / max(bh, 1) <= 4 && bw * bh >= 500 && !(Double(d.box.maxY) >= H - 5 && bw > W * 0.5)
        }
        let tracked = tracker.update(dets)
        // front-to-back, so closer boxes mask the ones behind them (pipeline.py occlusion masking)
        let ordered = tracked.map { ($0, roughDepth($0.det.box, frame)) }.sorted { $0.1 < $1.1 }.map { $0.0 }
        var rows: [PerceptionRow] = []
        for (id, det) in ordered {
            let tid = String(id)
            let occluders = rows.filter { $0.distanceM != nil }.map(\.box)
            let d = frame.depth.flatMap { medianDepth($0, box: det.box, frameSize: frame.size, exclude: occluders) }
            if let d {
                depthHistory[tid, default: []].append((frame.t, d))
                if depthHistory[tid]!.count > cfg.depth.velocity_window + 1 { depthHistory[tid]!.removeFirst() }
            }
            let cx = Double(det.box.midX)
            let bearing = atan((cx - frame.principalX) / frame.focalPx) * 180 / .pi
            rows.append(PerceptionRow(trackID: tid, className: Self.vehicleClass(cfg.classes[det.cls], det.box, d, frame.focalPx,
                                                                 Double(frame.size.width), cfg.detector),
                                      confidence: (det.conf * 1000).rounded() / 1000,
                                      distanceM: d.map { ($0 * 100).rounded() / 100 }, bearingDeg: bearing,
                                      velocityMs: Self.velocity(depthHistory[tid] ?? []),
                                      bboxAreaPx: Double(det.box.width * det.box.height), box: det.box))
        }
        let live = Set(tracked.map { String($0.id) })
        depthHistory = depthHistory.filter { live.contains($0.key) || ($0.value.last.map { frame.t - $0.0 < 10 } ?? false) }
        return rows
    }

    /// yolo_tracker.vehicle_class: a "car" narrower than any car (pinhole width at its depth) is a two-wheeler.
    static func vehicleClass(_ cls: String, _ box: CGRect, _ depth: Double?, _ focalPx: Double, _ width: Double,
                             _ c: CPEConfig.Detector) -> String {
        guard c.four_wheelers.contains(cls), let d = depth, d > 0 else { return cls }
        let x1 = Double(box.minX), x2 = Double(box.maxX)
        if x1 <= c.edge_margin_px || x2 >= width - c.edge_margin_px { return cls }
        return (x2 - x1) * d / focalPx < c.car_min_width_m ? "motorcycle" : cls
    }

    private func roughDepth(_ box: CGRect, _ f: FrameInput) -> Double {
        guard let dm = f.depth else { return .infinity }
        let x = Int(Double(box.midX) / Double(f.size.width) * Double(dm.width)).clamped(0, dm.width - 1)
        let y = Int(Double(box.midY) / Double(f.size.height) * Double(dm.height)).clamped(0, dm.height - 1)
        let v = Double(dm.at(x, y))
        return v.isFinite && v > 0 ? v : .infinity
    }

    /// depth_loader.median_depth_in_box: the central `center_frac` patch, closer boxes masked out,
    /// a 3x patch if it has no reading.
    func medianDepth(_ dm: DepthMap, box: CGRect, frameSize: CGSize, exclude: [CGRect]) -> Double? {
        let sx = Double(dm.width) / Double(frameSize.width), sy = Double(dm.height) / Double(frameSize.height)
        func scaled(_ b: CGRect) -> (Int, Int, Int, Int) {
            (Int(Double(b.minX) * sx), Int(Double(b.minY) * sy), Int(Double(b.maxX) * sx), Int(Double(b.maxY) * sy))
        }
        let (x1, y1, x2, y2) = scaled(box)
        let ex = exclude.map(scaled)
        let cx = (x1 + x2) / 2, cy = (y1 + y2) / 2
        let bw = max(1, Int(Double(x2 - x1) * cfg.depth.center_frac / 2))
        let bh = max(1, Int(Double(y2 - y1) * cfg.depth.center_frac / 2))
        for grow in [1, 3] {
            let rx1 = max(0, cx - bw * grow), rx2 = min(dm.width, cx + bw * grow)
            let ry1 = max(0, cy - bh * grow), ry2 = min(dm.height, cy + bh * grow)
            guard rx1 < rx2, ry1 < ry2 else { return nil }
            var vals: [Float] = []
            for y in ry1..<ry2 { for x in rx1..<rx2 {
                if ex.contains(where: { x >= $0.0 && x < $0.2 && y >= $0.1 && y < $0.3 }) { continue }
                let v = dm.at(x, y)
                if v.isFinite && Double(v) > cfg.depth.min_m && Double(v) < cfg.depth.max_m { vals.append(v) }
            } }
            if !vals.isEmpty {
                vals.sort()
                let m = vals.count
                return Double(m % 2 == 1 ? vals[m / 2] : (vals[m / 2 - 1] + vals[m / 2]) / 2)
            }
        }
        return nil
    }

    /// physics.compute_velocity: least-squares slope of depth over time, positive = approaching, >= 0.
    static func velocity(_ h: [(Double, Double)]) -> Double {
        guard h.count >= 2 else { return 0 }
        if h.count == 2 {
            let dt = h[1].0 - h[0].0
            return dt > 0 ? max(0, (h[0].1 - h[1].1) / dt) : 0
        }
        let n = Double(h.count)
        let mt = h.map(\.0).reduce(0, +) / n, md = h.map(\.1).reduce(0, +) / n
        let den = h.map { ($0.0 - mt) * ($0.0 - mt) }.reduce(0, +)
        guard den > 0 else { return 0 }
        let slope = h.map { ($0.0 - mt) * ($0.1 - md) }.reduce(0, +) / den
        return max(0, -slope)
    }
}

extension Comparable {
    func clamped(_ lo: Self, _ hi: Self) -> Self { min(max(self, lo), hi) }
}
