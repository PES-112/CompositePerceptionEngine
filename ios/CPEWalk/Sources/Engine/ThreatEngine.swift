import CoreML
import Foundation

// Ports of src/threat_engine/envelope.py, priority.py and engine.py.

struct Override { let track: FusedTrack; let ttcS: Double }

struct Arrival { let pArrive: Double; let expectedS: Double }

struct SceneContext { var inPath = false; var nearBoundary = false; var otherSurface = false }

struct RankedThreat { let track: FusedTrack; let priority: Double; let arrival: Arrival }

// MARK: - Safety envelope

struct SafetyEnvelope {
    let cfg: CPEConfig

    func inCorridor(lateral: Double, forward: Double, cls: String, width: Double) -> Bool {
        forward > 0 && abs(lateral) <= width / 2 + cfg.halfWidth(cls)
    }

    func check(_ t: FusedTrack) -> Override? {
        let e = cfg.envelope
        guard t.confirmed, t.misses <= e.max_coast_frames else { return nil }
        guard t.closingSpeedMs >= e.min_closing_ms, t.forwardM > 0 else { return nil }
        let ttc = t.distanceM / t.closingSpeedMs
        guard ttc <= e.ttc_s else { return nil }
        let lateralAtTTC = t.lateralM + t.relVelocity.x * ttc
        let w = e.envelope_corridor_m
        guard inCorridor(lateral: t.lateralM, forward: t.forwardM, cls: t.className, width: w)
                || abs(lateralAtTTC) <= w / 2 + cfg.halfWidth(t.className) else { return nil }
        return Override(track: t, ttcS: ttc)
    }

    func overrides(_ tracks: [FusedTrack]) -> [Override] {
        tracks.compactMap(check).sorted { ($0.ttcS, $0.track.distanceM) < ($1.ttcS, $1.track.distanceM) }
    }
}

// MARK: - GRU arrival predictor (priority.py GRUArrivalPredictor + samples.history_features)

/// Stage-1 history per track and the GRU's per-step features, exactly as
/// dataset/GRU/samples.history_features computes them.
final class GRUFeatures {
    struct Obs { let conf, bearing, dist, area, vel: Double }
    let history: Int
    private(set) var obs: [String: [Int: Obs]] = [:]
    private(set) var firstSeen: [String: Int] = [:]

    init(history: Int) { self.history = history }

    func observe(frame k: Int, tracks: [FusedTrack]) {
        for t in tracks {
            guard t.measured, let r = t.row, let d = r.distanceM else { continue }
            obs[t.trackID, default: [:]][k] = Obs(conf: r.confidence, bearing: r.bearingDeg, dist: d,
                                                  area: r.bboxAreaPx, vel: r.velocityMs)
            if firstSeen[t.trackID] == nil { firstSeen[t.trackID] = k }
        }
        let oldest = k - (history - 1)
        for id in Array(obs.keys) {
            obs[id] = obs[id]!.filter { $0.key >= oldest }
            if obs[id]!.isEmpty { obs[id] = nil; firstSeen[id] = nil }
        }
    }

    /// [history][9] for one track at frame k.
    func features(_ id: String, frame k: Int) -> [[Float]] {
        var x = Array(repeating: Array(repeating: Float(0), count: 9), count: history)
        guard let track = obs[id], let first = firstSeen[id] else { return x }
        var prev: (Int, Double, Double)?
        for i in 0..<history {
            let f = k - (history - 1 - i)
            guard let o = track[f] else { continue }
            var dDist = 0.0, dBearing = 0.0
            if let p = prev {
                let gap = Double(i - p.0)
                dDist = (o.dist - p.1) / gap
                dBearing = (o.bearing - p.2) / gap
            }
            let age = min(Double(f - first), 30)
            x[i] = [1, o.dist / 10, o.vel / 5, o.bearing / 35, dDist, dBearing / 10, o.conf,
                    log1p(max(o.area, 0)) / 15, age / 30].map(Float.init)
            prev = (i, o.dist, o.bearing)
        }
        return x
    }
}

final class GRUArrivalPredictor {
    private let model: MLModel
    private let cfg: CPEConfig
    let features: GRUFeatures
    private let classIndex: [String: Int]
    private let x: MLMultiArray
    private let cls: MLMultiArray

    init(config: CPEConfig, bundle: Bundle = .main) throws {
        cfg = config
        guard let url = bundle.url(forResource: "CPEArrivalGRU", withExtension: "mlmodelc") else {
            throw CPEError.missing("CPEArrivalGRU.mlpackage") }
        let mc = MLModelConfiguration()
        mc.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: mc)
        features = GRUFeatures(history: config.gru.history)
        classIndex = Dictionary(uniqueKeysWithValues: config.classes.enumerated().map { ($1, $0) })
        let n = config.gru.max_objects, t = config.gru.history
        x = try MLMultiArray(shape: [1, NSNumber(value: n), NSNumber(value: t), 9], dataType: .float32)
        cls = try MLMultiArray(shape: [1, NSNumber(value: n)], dataType: .int32)
    }

    /// The objects scored this frame, nearest first by Stage-1 distance, at most max_objects.
    func live(frame k: Int, tracks: [FusedTrack]) -> [FusedTrack] {
        tracks.filter { features.obs[$0.trackID]?[k] != nil }
            .sorted { ($0.row?.distanceM ?? .infinity) < ($1.row?.distanceM ?? .infinity) }
            .prefix(cfg.gru.max_objects).map { $0 }
    }

    func predict(frame k: Int, tracks: [FusedTrack]) throws -> [String: Arrival] {
        features.observe(frame: k, tracks: tracks)
        let scored = live(frame: k, tracks: tracks)
        guard !scored.isEmpty else { return [:] }
        let n = cfg.gru.max_objects, T = cfg.gru.history
        let xp = x.dataPointer.assumingMemoryBound(to: Float.self)
        xp.update(repeating: 0, count: n * T * 9)
        let cp = cls.dataPointer.assumingMemoryBound(to: Int32.self)
        cp.update(repeating: Int32(cfg.classes.count), count: n)          // UNKNOWN_CLASS for empty slots
        for (i, t) in scored.enumerated() {
            let f = features.features(t.trackID, frame: k)
            for s in 0..<T { for j in 0..<9 { xp[(i * T + s) * 9 + j] = f[s][j] } }
            cp[i] = Int32(classIndex[t.className] ?? cfg.classes.count)
        }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["x": x, "cls": cls]))
        guard let p = out.featureValue(for: "p")?.multiArrayValue else { throw CPEError.missing("GRU output p") }
        let K = cfg.gru.time_bins, H = Double(cfg.gru.horizon)
        let width = ceil(H / Double(K))
        let lo = (0..<K).map { Double($0) * width }
        let mids = (0..<K).map { i in (lo[i] + (i + 1 < K ? lo[i + 1] - 1 : H)) / 2 }
        let never = H + width
        var res: [String: Arrival] = [:]
        for (i, t) in scored.enumerated() {
            var probs = (0...K).map { p[[0, NSNumber(value: i), NSNumber(value: $0)] as [NSNumber]].doubleValue }
            let s = probs.reduce(0, +)
            if s > 0 { probs = probs.map { $0 / s } }
            let expected = zip(probs.prefix(K), mids).map(*).reduce(0, +) + probs[K] * never
            res[t.trackID] = Arrival(pArrive: 1 - probs[K], expectedS: expected * cfg.gru.step_s)
        }
        return res
    }
}

// MARK: - Fused priority

func contextMultiplier(_ c: SceneContext?, _ cfg: CPEConfig) -> Double {
    guard let c else { return 1 }
    var m = 1.0
    if c.inPath { m *= cfg.priority.context_in_path }
    if c.nearBoundary { m *= cfg.priority.context_near_boundary }
    if c.otherSurface { m *= cfg.priority.context_other_surface }
    return m
}

func rank(_ tracks: [FusedTrack], _ arrivals: [String: Arrival], contexts: [String: SceneContext], cfg: CPEConfig)
    -> [RankedThreat] {
    var out: [RankedThreat] = []
    for t in tracks {
        guard let a = arrivals[t.trackID] else { continue }
        let pr = a.pArrive * pow(cfg.severityOf(t.className), cfg.severity_gamma) * contextMultiplier(contexts[t.trackID], cfg)
        if pr >= cfg.priority.ignore_threshold { out.append(RankedThreat(track: t, priority: pr, arrival: a)) }
    }
    return out.sorted { ($0.priority, -$0.arrival.expectedS, -$0.track.distanceM) > ($1.priority, -$1.arrival.expectedS, -$1.track.distanceM) }
}

// MARK: - Engine: one call per processed frame (engine.py ThreatEngine.step)

struct EngineFrame {
    let k: Int
    let t: Double
    let tracks: [FusedTrack]
    let overrides: [Override]
    let ranked: [RankedThreat]
    let arrivals: [String: Arrival]
    let announcement: Announcement?
}

final class ThreatEngine {
    let cfg: CPEConfig
    let tracker: GroundTracker
    let predictor: GRUArrivalPredictor
    let announcer: AnnouncementManager
    private let envelope: SafetyEnvelope

    init(config: CPEConfig, predictor: GRUArrivalPredictor) {
        cfg = config
        tracker = GroundTracker(config: config)
        self.predictor = predictor
        announcer = AnnouncementManager(config: config.announcer)
        envelope = SafetyEnvelope(cfg: config)
    }

    func step(k: Int, t: Double, rows: [PerceptionRow], pose: EgoPose?, contexts: [String: SceneContext] = [:],
              edge: EdgeAlert? = nil, surface: String? = nil) throws -> EngineFrame {
        let tracks = tracker.step(t: t, rows: rows, pose: pose)
        let ov = envelope.overrides(tracks)
        let arrivals = try predictor.predict(frame: k, tracks: tracks)
        let ranked = rank(tracks.filter { $0.confirmed || $0.measured }, arrivals, contexts: contexts, cfg: cfg)
        let a = announcer.step(t: t, overrides: ov, ranked: ranked, edge: edge, surface: surface)
        return EngineFrame(k: k, t: t, tracks: tracks, overrides: ov, ranked: ranked, arrivals: arrivals, announcement: a)
    }
}
