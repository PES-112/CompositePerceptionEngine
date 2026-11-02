import CoreGraphics
import Foundation
import simd

// Phone-only street rules, added after the 2026-11-02 walks. They are not in the Python engine or the
// SANPO replay (SANPO has no phone LiDAR or ARKit floor), so no replay number covers them. Each one
// emits an EdgeAlert, so it goes through the announcer's edge lane: below the safety override,
// above object threats.
//
//   CrossingWatch     "Vehicles approaching from your left."  On the road or at its edge, from box
//                     motion alone. A vehicle beyond the LiDAR's ~5 m range has no depth, so fusion
//                     never tracks it.
//   PostWatch         "Bollard ahead, 2 meters. Watch out."  A detected bollard / hydrant / pole in
//                     the walking path.
//   PathObstacleScan  "Obstacle ahead, 2 meters."  Class-free: LiDAR points that stand up from the
//                     floor in the walking path (poles and pillars the detector misses).

enum StreetRules {
    static let vehicles: Set<String> = ["car", "bus", "truck", "motorcycle", "bicycle"]
    /// Typical height (m) per vehicle class, for a range guess from box height when LiDAR has none.
    static let vehicleHeightM: [String: Double] = ["car": 1.5, "bus": 3.0, "truck": 3.0, "motorcycle": 1.3, "bicycle": 1.3]
    static let posts: [String: String] = ["bollard": "bollard", "fire hydrant": "bollard", "pole": "pole"]
    static let pathHalfWidthM = 0.4
    /// Detections on the ground, not standing in the path: their boxes stay in the LiDAR scan.
    static let groundClasses: Set<String> = ["crosswalk", "pothole", "puddle", "stairs"]
}

// MARK: - Vehicles approaching while crossing

final class CrossingWatch {
    static let left = "vehicle_left", right = "vehicle_right", lefts = "vehicles_left", rights = "vehicles_right",
               both = "vehicles_both"

    struct Obs { let t: Double; let bearing: Double; let logArea: Double; let rangeM: Double }

    var windowS = 1.2, minSpanS = 0.6
    var sideDeg = 8.0               // |bearing| above this is "left" / "right"
    var inwardDegS = 3.0            // drift toward the image centre, head turns removed
    var growthPerS = 0.25           // box-area growth beyond what the walker's own motion explains
    var repeatS = 4.0
    private var hist: [String: [Obs]] = [:]
    private var lastSides = Set<Int>()            // -1 left, 1 right: what was last said
    private var lastAt = -Double.infinity
    private var lastPose: (Double, SIMD2<Double>)?
    private var walkSpeed = 0.0

    /// `headingDeg`: walker heading (clockwise, top view); a world-fixed object's image bearing drops by
    /// as much as the heading rises, so bearing + heading removes head turns.
    func step(t: Double, rows: [PerceptionRow], pose: EgoPose?, focalPx: Double, onRoad: Bool) -> EdgeAlert? {
        var heading = 0.0
        if let p = pose {
            heading = atan2(p.forward.x, p.forward.y) * 180 / .pi
            if let (t0, x0) = lastPose, t > t0 { walkSpeed = 0.5 * walkSpeed + 0.5 * simd_length(p.position - x0) / (t - t0) }
            lastPose = (t, p.position)
        }
        var live = Set<String>()
        for r in rows where StreetRules.vehicles.contains(r.className) && r.box.height > 1 {
            live.insert(r.trackID)
            let range = r.distanceM ?? focalPx * (StreetRules.vehicleHeightM[r.className] ?? 1.5) / Double(r.box.height)
            hist[r.trackID, default: []].append(Obs(t: t, bearing: r.bearingDeg + heading,
                                                    logArea: log(Double(r.box.width * r.box.height)), rangeM: range))
            hist[r.trackID]!.removeAll { t - $0.t > windowS }
        }
        hist = hist.filter { live.contains($0.key) }

        var nLeft = 0, nRight = 0
        for (_, h) in hist {
            guard h.count >= 3, let a = h.first, let b = h.last, b.t - a.t >= minSpanS else { continue }
            let bearingNow = b.bearing - heading
            guard abs(bearingNow) > sideDeg else { continue }
            let rate = slope(h.map { ($0.t, $0.bearing) })
            // a static object grows by 2 v cos(bearing) / range per second as the walker closes in
            let ownGrowth = 2 * walkSpeed * cos(bearingNow * .pi / 180) / max(b.rangeM, 1)
            let growth = slope(h.map { ($0.t, $0.logArea) }) - ownGrowth
            let inward = bearingNow < 0 ? rate >= inwardDegS : rate <= -inwardDegS
            guard inward || growth >= growthPerS else { continue }
            if bearingNow < 0 { nLeft += 1 } else { nRight += 1 }
        }
        guard onRoad, nLeft + nRight > 0 else {
            if nLeft + nRight == 0 { lastSides = [] }
            return nil
        }
        let sides = Set((nLeft > 0 ? [-1] : []) + (nRight > 0 ? [1] : []))
        // speak when a side is new, else repeat every repeatS while vehicles keep coming
        guard !sides.isSubset(of: lastSides) || t - lastAt >= repeatS else { return nil }
        let kind = sides.count == 2 ? Self.both
            : nLeft > 0 ? (nLeft > 1 ? Self.lefts : Self.left) : (nRight > 1 ? Self.rights : Self.right)
        lastSides = sides; lastAt = t
        return EdgeAlert(t: t, kind: kind, distanceM: 0)
    }

    private func slope(_ p: [(Double, Double)]) -> Double {
        let n = Double(p.count), mt = p.map(\.0).reduce(0, +) / n, mv = p.map(\.1).reduce(0, +) / n
        let den = p.map { ($0.0 - mt) * ($0.0 - mt) }.reduce(0, +)
        return den > 0 ? p.map { ($0.0 - mt) * ($0.1 - mv) }.reduce(0, +) / den : 0
    }
}

// MARK: - Detected posts in the path

final class PostWatch {
    var maxM = 4.0
    private var said = Set<String>()

    func step(t: Double, rows: [PerceptionRow]) -> EdgeAlert? {
        let inPath = rows.compactMap { r -> (PerceptionRow, String, Double)? in
            guard let word = StreetRules.posts[r.className], let d = r.distanceM, d <= maxM, !said.contains(r.trackID),
                  abs(d * sin(r.bearingDeg * .pi / 180)) <= StreetRules.pathHalfWidthM + 0.1 else { return nil }
            return (r, word, d)
        }
        guard let (r, word, d) = inPath.min(by: { $0.2 < $1.2 }) else { return nil }
        said.insert(r.trackID)
        return EdgeAlert(t: t, kind: word, distanceM: d)
    }
}

// MARK: - Class-free obstacle from LiDAR

/// What PathObstacleScan needs to put a portrait depth pixel in the world (filled from the ARFrame).
struct DepthGeometry {
    let fx, fy, cx, cy: Double             // sensor intrinsics at depth-map resolution (landscape)
    let sensorHeight: Int                  // landscape depth-map height; portrait u = sensorHeight - 1 - y
    let rotation: simd_double3x3           // camera -> world (ARKit, y up)
    let position: SIMD3<Double>            // camera position in the world
}

struct PathPoint { let forward: Double; let lateral: Double; let height: Double }   // height above the floor

enum PathObstacleScan {
    static let maxForwardM = 3.0, minHeightM = 0.2, maxHeightM = 2.0
    static let minPoints = 15, clusterDepthM = 0.3, minSpanM = 0.3

    /// Points in the walker frame, with the floor taken as the 10th-percentile height of the points
    /// 0.5–3 m ahead. Pixels inside `exclude` (boxes of objects the detector already names) are skipped.
    static func points(_ d: DepthMap, _ g: DepthGeometry, pose: EgoPose, exclude: [CGRect]) -> [PathPoint] {
        var raw: [(Double, Double, Double)] = []          // forward, lateral, world y
        var floorY: [Double] = []
        for v in stride(from: 0, to: d.height, by: 2) { for u in stride(from: 0, to: d.width, by: 2) {
            let z = Double(d.at(u, v))
            guard z.isFinite, z > 0.3, z < 4.5 else { continue }
            let x = Double(v), y = Double(g.sensorHeight - 1 - u)
            let cam = SIMD3((x - g.cx) / g.fx * z, -(y - g.cy) / g.fy * z, -z)
            let w = g.rotation * cam + g.position
            let top = SIMD2(w.x - g.position.x, -(w.z - g.position.z))
            let fwd = simd_dot(top, pose.forward), lat = simd_dot(top, pose.right)
            if fwd > 0.5 && fwd < 3 && abs(lat) < 1.5 { floorY.append(w.y) }
            guard fwd > 0.3, fwd <= maxForwardM, abs(lat) <= StreetRules.pathHalfWidthM else { continue }
            let p = CGPoint(x: u, y: v)
            if exclude.contains(where: { $0.contains(p) }) { continue }
            raw.append((fwd, lat, w.y))
        }}
        guard floorY.count >= 50 else { return [] }
        floorY.sort()
        let floor = floorY[floorY.count / 10]
        let camHeight = g.position.y - floor
        guard camHeight > 0.7, camHeight < 2.2 else { return [] }   // no floor in view
        return raw.map { PathPoint(forward: $0.0, lateral: $0.1, height: $0.2 - floor) }
    }

    /// The distance of the nearest thing standing up in the path: enough points 0.2–2 m above the floor,
    /// spanning ≥ 0.3 m of height within 0.3 m of depth (a step or curb is one short riser).
    static func nearest(_ pts: [PathPoint]) -> Double? {
        let up = pts.filter { $0.height >= minHeightM && $0.height <= maxHeightM }.sorted { $0.forward < $1.forward }
        guard up.count >= minPoints else { return nil }
        var i = 0
        while i + minPoints <= up.count {
            let d0 = up[i + 2].forward                                   // ignore a couple of stray points
            let cluster = up[i...].prefix { $0.forward <= d0 + clusterDepthM }
            if cluster.count >= minPoints, let lo = cluster.map(\.height).min(), let hi = cluster.map(\.height).max(),
               hi - lo >= minSpanM {
                return median(cluster.map(\.forward))
            }
            i += max(1, cluster.count / 2)
        }
        return nil
    }
}

/// "Obstacle ahead" once per episode: confirmed on 2 scans in a row, re-armed after 2 s clear.
final class PathObstacleAlerter {
    var confirm = 2, clearS = 2.0
    private var run = 0, armed = true, lastSeen = -Double.infinity

    func step(t: Double, distanceM: Double?) -> EdgeAlert? {
        guard let d = distanceM else {
            run = 0
            if t - lastSeen >= clearS { armed = true }
            return nil
        }
        lastSeen = t
        run += 1
        guard armed, run >= confirm else { return nil }
        armed = false
        return EdgeAlert(t: t, kind: "obstacle", distanceM: d)
    }
}
