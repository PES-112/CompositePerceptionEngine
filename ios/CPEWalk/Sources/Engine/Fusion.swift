import Foundation
import simd

/// Camera pose on the ground plane (src/threat_engine/fusion.py EgoPose): position and the camera's
/// right / forward unit vectors in a world-fixed top view. On the phone it comes from ARKit.
struct EgoPose {
    var position: SIMD2<Double>
    var right: SIMD2<Double>
    var forward: SIMD2<Double>

    static let identity = EgoPose(position: [0, 0], right: [1, 0], forward: [0, 1])

    /// Columns: right, forward.
    var basis: simd_double2x2 { simd_double2x2(columns: (right, forward)) }
    func toWorld(x: Double, z: Double) -> SIMD2<Double> { position + basis * SIMD2(x, z) }
}

/// One perceived object in one frame (the Stage-1 row fields fusion and the GRU need).
struct PerceptionRow {
    var trackID: String
    var className: String
    var confidence: Double
    var distanceM: Double?
    var bearingDeg: Double
    var velocityMs: Double
    var bboxAreaPx: Double
    var box: CGRect = .zero            // portrait image pixels; for the overlay and scene context
}

/// One track after fusion, relative to the walker and their walking direction (fusion.py FusedTrack).
struct FusedTrack {
    var trackID: String
    var className: String
    var confidence: Double
    var forwardM: Double
    var lateralM: Double
    var distanceM: Double
    var bearingDeg: Double
    var closingSpeedMs: Double
    var lateralSpeedMs: Double
    var relVelocity: SIMD2<Double>      // (lateral, forward)
    var positionStdM: Double
    var hits: Int
    var misses: Int
    var confirmed: Bool
    var measured: Bool
    var row: PerceptionRow?
}

/// Constant-velocity Kalman filter per track on the world ground plane, with the walker's motion
/// removed using the camera pose: a line-by-line port of fusion.py GroundTracker, with its fitted
/// noise model (Student-t update) from cpe_config.json.
final class GroundTracker {
    private struct Filter {
        var x: SIMD4<Double>            // X, Z, VX, VZ (world)
        var P: simd_double4x4
        var className: String
        var confidence: Double
        var hits = 1
        var misses = 0
        var gated = 0
        var row: PerceptionRow?
        var measured = true
    }

    private let noise: CPEConfig.Noise
    private let cfg: CPEConfig.Fusion
    private let staticClasses: Set<String>
    private var filters: [String: Filter] = [:]
    private var order: [String] = []    // insertion order, as Python dicts iterate
    private var lastT: Double?
    private var lastPose: EgoPose?
    private(set) var egoVelocity = SIMD2<Double>(0, 0)

    init(config: CPEConfig) {
        noise = config.noise
        cfg = config.fusion
        staticClasses = Set(config.static_classes)
    }

    func walkingBasis(_ pose: EgoPose) -> simd_double2x2 {
        let speed = simd_length(egoVelocity)
        if speed < cfg.min_walk_speed { return pose.basis }
        let fwd = egoVelocity / speed
        let cam = pose.basis
        let det = cam.columns.0.x * cam.columns.1.y - cam.columns.1.x * cam.columns.0.y
        let sign: Double = det > 0 ? 1 : -1
        let right = sign * SIMD2(fwd.y, -fwd.x)
        return simd_double2x2(columns: (right, fwd))
    }

    func step(t: Double, rows: [PerceptionRow], pose: EgoPose?) -> [FusedTrack] {
        let pose = pose ?? .identity
        let dt = lastT.map { max(0, t - $0) } ?? 0
        updateEgo(pose, dt: dt)
        for id in order { predict(&filters[id]!, dt: dt); filters[id]!.measured = false }

        for r in rows {
            guard let d = r.distanceM, d.isFinite, d > 0 else { continue }   // coast on the prediction
            let (z, R) = measure(depth: d, bearingDeg: r.bearingDeg, pose: pose)
            guard var f = filters[r.trackID] else {
                filters[r.trackID] = newFilter(z, R, r)
                order.append(r.trackID)
                continue
            }
            if !update(&f, z, R) {
                f.gated += 1
                if f.gated >= cfg.max_gated { filters[r.trackID] = newFilter(z, R, r) } else { filters[r.trackID] = f }
                continue
            }
            f.hits += 1; f.misses = 0; f.gated = 0; f.measured = true
            f.className = r.className; f.confidence = r.confidence; f.row = r
            filters[r.trackID] = f
        }
        for id in order where !(filters[id]?.measured ?? true) {
            filters[id]!.misses += 1
            if filters[id]!.misses > cfg.max_misses { filters[id] = nil }
        }
        order.removeAll { filters[$0] == nil }
        lastT = t
        lastPose = pose
        return order.map { relative($0, filters[$0]!, pose) }
    }

    private func updateEgo(_ pose: EgoPose, dt: Double) {
        guard let last = lastPose, dt > 0 else { return }
        let sample = (pose.position - last.position) / dt
        egoVelocity = cfg.ego_smoothing * sample + (1 - cfg.ego_smoothing) * egoVelocity
    }

    private func measure(depth: Double, bearingDeg: Double, pose: EgoPose) -> (SIMD2<Double>, simd_double2x2) {
        let theta = bearingDeg * .pi / 180
        let xCam = depth * tan(theta), zCam = depth
        let sdDepth = noise.depthStd(depth)
        let sdLat = hypot(depth * noise.bearing_std_deg * .pi / 180, sdDepth * abs(tan(theta)))
        let B = pose.basis
        let R = B * simd_double2x2(diagonal: SIMD2(sdLat * sdLat, sdDepth * sdDepth)) * B.transpose
        return (pose.toWorld(x: xCam, z: zCam), R)
    }

    private func newFilter(_ z: SIMD2<Double>, _ R: simd_double2x2, _ r: PerceptionRow) -> Filter {
        var P = simd_double4x4(0)
        P[0][0] = R[0][0]; P[0][1] = R[0][1]; P[1][0] = R[1][0]; P[1][1] = R[1][1]
        let s2 = noise.init_speed_std * noise.init_speed_std
        P[2][2] = s2; P[3][3] = s2
        return Filter(x: SIMD4(z.x, z.y, 0, 0), P: P, className: r.className, confidence: r.confidence, row: r)
    }

    private func predict(_ f: inout Filter, dt: Double) {
        guard dt > 0 else { return }
        var F = matrix_identity_double4x4
        F[2][0] = dt; F[3][1] = dt                      // column-major: x += vx·dt, z += vz·dt
        let a = staticClasses.contains(f.className) ? noise.static_accel_std : noise.accel_std
        let q = a * a
        var Q = simd_double4x4(0)
        let q11 = pow(dt, 4) / 4 * q, q12 = pow(dt, 3) / 2 * q, q22 = dt * dt * q
        Q[0][0] = q11; Q[2][0] = q12; Q[0][2] = q12; Q[2][2] = q22
        Q[1][1] = q11; Q[3][1] = q12; Q[1][3] = q12; Q[3][3] = q22
        f.x = F * f.x
        f.P = F * f.P * F.transpose + Q
    }

    /// Kalman update; false (no change) if the measurement fails the gate.
    private func update(_ f: inout Filter, _ z: SIMD2<Double>, _ Rin: simd_double2x2) -> Bool {
        let Ppos = simd_double2x2(columns: (SIMD2(f.P[0][0], f.P[0][1]), SIMD2(f.P[1][0], f.P[1][1])))
        let y = z - SIMD2(f.x.x, f.x.y)
        var S = Ppos + Rin
        var Sinv = S.inverse
        let nis = simd_dot(y, Sinv * y)
        if nis > noise.gate_nis { return false }
        var R = Rin
        if noise.robust_nu > 0 {
            R = Rin * ((noise.robust_nu + nis) / (noise.robust_nu + 2))
            S = Ppos + R
            Sinv = S.inverse
        }
        // K = P Hᵀ S⁻¹ : 4×2. Hᵀ picks P's first two columns.
        let PH0 = f.P.columns.0, PH1 = f.P.columns.1        // columns 0, 1 of P (4-vectors)
        let K0 = PH0 * Sinv[0][0] + PH1 * Sinv[0][1]        // column 0 of K
        let K1 = PH0 * Sinv[1][0] + PH1 * Sinv[1][1]        // column 1 of K
        f.x = f.x + K0 * y.x + K1 * y.y
        // Joseph form: (I - KH) P (I - KH)ᵀ + K R Kᵀ
        var KH = simd_double4x4(0)
        KH.columns.0 = K0; KH.columns.1 = K1
        let IKH = matrix_identity_double4x4 - KH
        var KRK = simd_double4x4(0)
        for i in 0..<4 { for j in 0..<4 {
            let ki = SIMD2(K0[i], K1[i]), kj = SIMD2(K0[j], K1[j])
            KRK[j][i] = simd_dot(ki, R * kj)
        } }
        f.P = IKH * f.P * IKH.transpose + KRK
        return true
    }

    private func relative(_ id: String, _ f: Filter, _ pose: EgoPose) -> FusedTrack {
        let W = walkingBasis(pose)
        let p = W.transpose * (SIMD2(f.x.x, f.x.y) - pose.position)      // (lateral, forward)
        let v = W.transpose * (SIMD2(f.x.z, f.x.w) - egoVelocity)
        let dist = simd_length(p)
        let closing = dist > 1e-6 ? -simd_dot(p, v) / dist : 0
        // trace of Wᵀ P_pos W, as fusion.py. Not simply trace(P_pos): when standing, W is the camera's
        // right / forward pair, which need not be exactly perpendicular.
        let Ppos = simd_double2x2(columns: (SIMD2(f.P[0][0], f.P[0][1]), SIMD2(f.P[1][0], f.P[1][1])))
        let posCov = W.transpose * Ppos * W
        let posTrace = posCov[0][0] + posCov[1][1]
        return FusedTrack(trackID: id, className: f.className, confidence: f.confidence, forwardM: p.y, lateralM: p.x,
                          distanceM: dist, bearingDeg: atan2(p.x, p.y) * 180 / .pi, closingSpeedMs: closing,
                          lateralSpeedMs: v.x, relVelocity: v, positionStdM: sqrt(max(posTrace, 0)),
                          hits: f.hits, misses: f.misses, confirmed: f.hits >= cfg.min_hits, measured: f.measured,
                          row: f.row)
    }
}
