import CoreImage
import CoreML
import Foundation

// Walkable-surface segmentation on the phone: ports of src/segmentation/context.py (walker surface,
// edge ahead, per-object ground and SceneContext) and edge_alert.py (EdgeAlerter, SurfaceTracker).
// Class ids: 0 walkable, 1 crosswalk, 2 road, 3 curb, 4 stairs, 5 non-walkable ground, 6 obstacle, 7 sky.

enum Ground {
    static let walkable: Set<UInt8> = [0, 1]
    static let unsafe: Set<UInt8> = [2, 3, 4, 5]
    static let all: Set<UInt8> = walkable.union(unsafe)
    static let road: UInt8 = 2, curb: UInt8 = 3, stairs: UInt8 = 4, nonWalkable: UInt8 = 5
}

/// A class map on a coarse grid of the upright image (row-major).
struct ClassMap {
    let width: Int
    let height: Int
    var ids: [UInt8]
    func at(_ x: Int, _ y: Int) -> UInt8 { ids[y * width + x] }
}

struct SceneFrame {
    var walkerSurface: UInt8?
    var edge: (kind: UInt8, distanceM: Double)?
    var contexts: [String: SceneContext] = [:]
}

final class Segmenter {
    private let model: MLModel
    private let inputName: String
    private let size: Int
    private let grid: (Int, Int)
    private let s: CPEConfig.Segmentation
    private let ciContext = CIContext()
    private var buffer: CVPixelBuffer?

    init(config: CPEConfig, bundle: Bundle = .main, grid: (Int, Int) = (180, 240)) throws {
        guard let url = bundle.url(forResource: "CPESegmenter", withExtension: "mlmodelc") else {
            throw CPEError.missing("CPESegmenter.mlpackage") }
        let mc = MLModelConfiguration()
        mc.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: mc)
        inputName = model.modelDescription.inputDescriptionsByName.first { $0.value.type == .image }?.key ?? "image"
        size = 640
        self.grid = grid
        s = config.segmentation
    }

    /// The class map of an upright frame, sampled onto `grid` (width x height) cells. The model gives
    /// stride-8 class logits; each cell takes them bilinearly (as the model's own upsample,
    /// align_corners = false), adds context.ROAD_LOGIT_BIAS to road, and keeps the best class.
    func classMap(_ frame: FrameInput) throws -> ClassMap? {
        guard let (pb, r, pad) = Detector.letterbox(frame.image, size: size, context: ciContext, into: &buffer) else { return nil }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pb)]))
        guard let name = out.featureNames.first, let a = out.featureValue(for: name)?.multiArrayValue,
              a.shape.count == 4 else { return nil }
        let nc = a.shape[1].intValue, lh = a.shape[2].intValue, lw = a.shape[3].intValue
        let st = a.strides.map(\.intValue)
        let scale = Double(size) / Double(lw)
        let (gw, gh) = grid
        var ids = [UInt8](repeating: 255, count: gw * gh)
        let W = Double(frame.size.width), H = Double(frame.size.height)
        let bias = s.road_logit_bias, road = Int(Ground.road)
        func run(_ v: (Int) -> Float) {
            var best = [Double](repeating: 0, count: nc)
            for gy in 0..<gh { for gx in 0..<gw {
                let lx = ((Double(gx) + 0.5) / Double(gw) * W) * r + pad.x
                let ly = ((Double(gy) + 0.5) / Double(gh) * H) * r + pad.y
                let u = min(max(lx / scale - 0.5, 0), Double(lw - 1)), w = min(max(ly / scale - 0.5, 0), Double(lh - 1))
                let u0 = Int(u), w0 = Int(w), u1 = min(u0 + 1, lw - 1), w1 = min(w0 + 1, lh - 1)
                let fu = u - Double(u0), fw = w - Double(w0)
                for c in 0..<nc {
                    let b = c * st[1]
                    let top = Double(v(b + w0 * st[2] + u0 * st[3])) * (1 - fu) + Double(v(b + w0 * st[2] + u1 * st[3])) * fu
                    let bot = Double(v(b + w1 * st[2] + u0 * st[3])) * (1 - fu) + Double(v(b + w1 * st[2] + u1 * st[3])) * fu
                    best[c] = top * (1 - fw) + bot * fw + (c == road ? bias : 0)
                }
                var k = 0
                for c in 1..<nc where best[c] > best[k] { k = c }
                ids[gy * gw + gx] = UInt8(k)
            } }
        }
        switch a.dataType {
        case .float16:
            let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
            run { Float(p[$0]) }
        case .float32:
            let p = a.dataPointer.assumingMemoryBound(to: Float32.self)
            run { p[$0] }
        default:
            run { a[$0].floatValue }
        }
        return ClassMap(width: gw, height: gh, ids: ids)
    }

    // MARK: context.py

    private func majority(_ c: [UInt8]) -> UInt8? {
        guard c.count >= s.min_ground_px else { return nil }
        var n = [Int](repeating: 0, count: 256)
        for v in c { n[Int(v)] += 1 }
        return UInt8(n.indices.max { n[$0] < n[$1] || (n[$0] == n[$1] && $0 > $1) }!)   // ties -> lowest id, as argmax
    }

    func walkerSurface(_ m: ClassMap) -> UInt8? {
        let y0 = Int(s.walker_rows[0] * Double(m.height)), y1 = Int(s.walker_rows[1] * Double(m.height))
        let x0 = Int(s.walker_cols[0] * Double(m.width)), x1 = Int(s.walker_cols[1] * Double(m.width))
        var ground: [UInt8] = [], total = 0
        for y in y0..<y1 { for x in x0..<x1 { total += 1; let v = m.at(x, y); if Ground.all.contains(v) { ground.append(v) } } }
        return Double(ground.count) >= 0.05 * Double(total) ? majority(ground) : nil
    }

    /// Depth of a class-map cell, from the (any-resolution) depth map.
    private func depthAt(_ d: DepthMap, _ m: ClassMap, _ x: Int, _ y: Int) -> Float {
        d.at((x * d.width / m.width).clamped(0, d.width - 1), (y * d.height / m.height).clamped(0, d.height - 1))
    }

    func edgeAhead(_ m: ClassMap, _ d: DepthMap, walker: UInt8?) -> (UInt8, Double)? {
        guard let w = walker, Ground.walkable.contains(w) else { return nil }
        let x0 = Int(s.edge_cols[0] * Double(m.width)), x1 = Int(s.edge_cols[1] * Double(m.width))
        for y in stride(from: m.height - 1, through: 0, by: -1) {
            var cls: [UInt8] = [], dep: [Float] = []
            for x in x0..<x1 { let v = m.at(x, y); if Ground.all.contains(v) { cls.append(v); dep.append(depthAt(d, m, x, y)) } }
            if cls.count < s.min_ground_px { continue }
            guard let kind = majority(cls) else { continue }
            if Ground.unsafe.contains(kind) {
                let dist = zip(cls, dep).filter { $0.0 == kind && $0.1.isFinite && $0.1 > 0 }.map { Double($0.1) }
                if let med = median(dist), med <= s.max_edge_m { return (kind, med) }
                return nil
            }
            if let med = median(dep.filter { $0.isFinite && $0 > 0 }.map(Double.init)), med > s.max_edge_m { return nil }
        }
        return nil
    }

    /// Ground cells in the object's columns at about its depth (context.object_ground).
    private func objectGround(_ m: ClassMap, _ d: DepthMap, cxFrac: Double, distance: Double, halfWidthFrac: Double,
                              widen: Double) -> [UInt8] {
        let hw = max(1, Int((halfWidthFrac * widen * Double(m.width)).rounded()))
        let c = Int((cxFrac * Double(m.width)).rounded())
        let lo = max(0, c - hw), hi = min(m.width, c + hw + 1)
        guard lo < hi else { return [] }
        let tol = max(s.depth_tol_min_m, s.depth_tol_frac * distance)
        var out: [UInt8] = []
        for y in 0..<m.height { for x in lo..<hi {
            let v = m.at(x, y)
            guard Ground.all.contains(v) else { continue }
            let z = Double(depthAt(d, m, x, y))
            if z.isFinite && abs(z - distance) <= tol { out.append(v) }
        } }
        return out
    }

    func sceneFrame(_ m: ClassMap, _ d: DepthMap?, rows: [PerceptionRow], imageWidth: Double) -> SceneFrame {
        let walker = walkerSurface(m)
        guard let d else { return SceneFrame(walkerSurface: walker, edge: nil) }
        var out = SceneFrame(walkerSurface: walker, edge: edgeAhead(m, d, walker: walker))
        let walkerOK = walker.map(Ground.walkable.contains) ?? false
        for r in rows {
            guard let dist = r.distanceM, r.box.width > 0 else { continue }
            let cx = Double(r.box.midX) / imageWidth, hwf = max(0.005, Double(r.box.width) / 2 / imageWidth)
            let surface = majority(objectGround(m, d, cxFrac: cx, distance: dist, halfWidthFrac: hwf, widen: 1))
            let wide = objectGround(m, d, cxFrac: cx, distance: dist, halfWidthFrac: hwf, widen: 2)
            let n = Double(max(1, wide.count))
            let safe = Double(wide.filter(Ground.walkable.contains).count) / n
            let unsafe = Double(wide.filter(Ground.unsafe.contains).count) / n
            let near = wide.count >= s.min_ground_px && (Double(wide.filter { $0 == Ground.curb }.count) / n >= s.curb_share
                                                         || (safe >= s.boundary_share && unsafe >= s.boundary_share))
            out.contexts[r.trackID] = SceneContext(
                inPath: walkerOK && surface.map(Ground.walkable.contains) == true,
                nearBoundary: near,
                otherSurface: walkerOK && (surface == Ground.road || surface == Ground.nonWalkable))
        }
        return out
    }

    func label(_ id: UInt8) -> String { s.labels[String(id)] ?? "ground" }
}

func median(_ v: [Double]) -> Double? {
    guard !v.isEmpty else { return nil }
    let s = v.sorted(), n = s.count
    return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
}

// MARK: edge_alert.py

/// "Curb ahead": curb, stairs or road within alert_m in confirm consecutive segmentation frames,
/// once per episode (re-armed after clear_s with no announced edge).
final class EdgeAlerter {
    private let kinds: Set<UInt8> = [Ground.curb, Ground.stairs, Ground.road]
    private let alertM: Double, confirm: Int, clearS: Double
    private var armed = true, lastSeen = -Double.infinity, runKind: UInt8?, run = 0

    init(_ s: CPEConfig.Segmentation) { alertM = s.edge_alert_m; confirm = s.edge_confirm_frames; clearS = s.edge_clear_s }

    func step(t: Double, edge: (kind: UInt8, distanceM: Double)?, label: (UInt8) -> String) -> EdgeAlert? {
        let announced = edge.map { kinds.contains($0.kind) } ?? false
        if announced { lastSeen = t } else if t - lastSeen >= clearS { armed = true }
        let near = announced && edge!.distanceM <= alertM
        if near && edge!.kind == runKind { run += 1 } else if near { runKind = edge!.kind; run = 1 } else { runKind = nil; run = 0 }
        if armed && run >= confirm, let e = edge {
            armed = false
            return EdgeAlert(t: t, kind: label(e.kind), distanceM: e.distanceM)
        }
        return nil
    }
}

/// The walker's surface for narration, debounced over `confirm` consecutive frames.
final class SurfaceTracker {
    private let confirm: Int
    private var current: UInt8?, candidate: UInt8?, run = 0
    init(confirm: Int) { self.confirm = confirm }

    func step(_ walker: UInt8?) -> UInt8? {
        if let w = walker, w != current {
            run = w == candidate ? run + 1 : 1
            candidate = w
            if run >= confirm { current = w; candidate = nil; run = 0 }
        } else if walker != nil {
            candidate = nil; run = 0
        }
        return current
    }
}
