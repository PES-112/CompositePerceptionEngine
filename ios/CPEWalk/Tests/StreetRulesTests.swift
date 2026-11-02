import XCTest
@testable import CPEWalk

/// The phone-only street rules (StreetRules.swift) and their phrases.
final class StreetRulesTests: XCTestCase {
    private var cfg: CPEConfig!
    private let fx = 1400.0, width = 1440.0

    override func setUpWithError() throws { cfg = try CPEConfig.load(bundle: Bundle(for: WalkPipeline.self)) }

    /// A car-sized box at `bearing` degrees, `heightPx` tall, no LiDAR depth (beyond ~5 m).
    private func car(_ id: String, bearing: Double, heightPx: Double, cls: String = "car") -> PerceptionRow {
        let cx = width / 2 + tan(bearing * .pi / 180) * fx
        let box = CGRect(x: cx - heightPx, y: 900, width: 2 * heightPx, height: heightPx)
        return PerceptionRow(trackID: id, className: cls, confidence: 0.8, distanceM: nil, bearingDeg: bearing,
                             velocityMs: 0, bboxAreaPx: Double(box.width * box.height), box: box)
    }

    private func pose(x: Double = 0, y: Double = 0, headingDeg: Double = 0) -> EgoPose {
        let h = headingDeg * .pi / 180
        let f = SIMD2(sin(h), cos(h))
        return EgoPose(position: [x, y], right: [f.y, -f.x], forward: f)
    }

    /// 2 s at 5 Hz; `row(i)` and `p(i)` give frame i's car and pose.
    private func run(_ w: CrossingWatch, onRoad: Bool, row: (Int) -> PerceptionRow, p: (Int) -> EgoPose) -> [String] {
        (0..<10).compactMap { i in
            w.step(t: Double(i) * 0.2, rows: [row(i)], pose: p(i), focalPx: fx, onRoad: onRoad)?.kind
        }
    }

    func testCarFromTheLeftWhileCrossing() {
        let out = run(CrossingWatch(), onRoad: true, row: { self.car("1", bearing: -30 + 3 * Double($0), heightPx: 120) },
                      p: { _ in self.pose() })
        XCTAssertEqual(out, [CrossingWatch.left])
    }

    func testOnlyOnTheRoad() {
        let out = run(CrossingWatch(), onRoad: false, row: { self.car("1", bearing: -30 + 3 * Double($0), heightPx: 120) },
                      p: { _ in self.pose() })
        XCTAssertTrue(out.isEmpty)
    }

    func testGrowingFromTheRightAndBothSides() {
        // straight at the walker from the right: fixed bearing, box growing ~50%/s
        var out = run(CrossingWatch(), onRoad: true, row: { self.car("1", bearing: 25, heightPx: 80 * pow(1.045, Double($0))) },
                      p: { _ in self.pose() })
        XCTAssertEqual(out, [CrossingWatch.right])
        let w = CrossingWatch()
        out = (0..<10).compactMap { i in
            w.step(t: Double(i) * 0.2, rows: [car("1", bearing: -30 + 3 * Double(i), heightPx: 120),
                                              car("2", bearing: 35 - 3 * Double(i), heightPx: 100, cls: "motorcycle")],
                   pose: pose(), focalPx: fx, onRoad: true)?.kind
        }
        XCTAssertEqual(out, [CrossingWatch.both])
    }

    func testNoAlertForParkedCarsHeadTurnsOrCarsLeaving() {
        // parked car, walker turning the head right at 15°/s: image bearing falls, world bearing fixed
        var out = run(CrossingWatch(), onRoad: true, row: { self.car("1", bearing: -20 - 3 * Double($0), heightPx: 120) },
                      p: { self.pose(headingDeg: 3 * Double($0)) })
        XCTAssertTrue(out.isEmpty, "head turn")
        // parked car 8 m ahead-left while the walker crosses at 1.3 m/s: growth explained by walking
        out = run(CrossingWatch(), onRoad: true, row: { i in
            let d = 8 - 0.26 * Double(i)
            return self.car("1", bearing: -12, heightPx: self.fx * 1.5 / d)
        }, p: { self.pose(y: 0.26 * Double($0)) })
        XCTAssertTrue(out.isEmpty, "parked car")
        // a car that has passed, moving away to the left
        out = run(CrossingWatch(), onRoad: true, row: { self.car("1", bearing: -15 - 3 * Double($0), heightPx: 120) },
                  p: { _ in self.pose() })
        XCTAssertTrue(out.isEmpty, "leaving")
    }

    func testPostInThePath() {
        let w = PostWatch()
        func post(_ id: String, _ cls: String, d: Double, bearing: Double) -> PerceptionRow {
            PerceptionRow(trackID: id, className: cls, confidence: 0.6, distanceM: d, bearingDeg: bearing, velocityMs: 0,
                          bboxAreaPx: 4000, box: CGRect(x: 700, y: 900, width: 40, height: 100))
        }
        XCTAssertNil(w.step(t: 0, rows: [post("1", "fire hydrant", d: 3, bearing: 30)]))       // off the path
        let a = w.step(t: 0.2, rows: [post("2", "fire hydrant", d: 2.4, bearing: 3), post("3", "pole", d: 3.5, bearing: -2)])
        XCTAssertEqual(a?.kind, "bollard")
        XCTAssertEqual(w.step(t: 0.4, rows: [post("2", "fire hydrant", d: 2.2, bearing: 3), post("3", "pole", d: 3.3, bearing: -2)])?.kind,
                       "pole")
        XCTAssertNil(w.step(t: 0.6, rows: [post("2", "fire hydrant", d: 2.0, bearing: 3)]))   // once per object
    }

    func testLidarObstacle() {
        var floor: [PathPoint] = []
        for f in stride(from: 0.4, through: 3.0, by: 0.05) { for l in stride(from: -0.4, through: 0.4, by: 0.1) {
            floor.append(PathPoint(forward: f, lateral: l, height: 0))
        } }
        XCTAssertNil(PathObstacleScan.nearest(floor))
        // a pole 2.1 m ahead, 0 to 1.9 m tall
        let pole = stride(from: 0.0, through: 1.9, by: 0.05).map { PathPoint(forward: 2.1, lateral: 0.05, height: $0) }
        XCTAssertEqual(PathObstacleScan.nearest(floor + pole) ?? 0, 2.1, accuracy: 0.05)
        // steps up: 0.15 m risers every 0.3 m, from 1 m ahead
        var steps = floor.filter { $0.forward < 1 }
        for i in 0..<7 { for l in stride(from: -0.4, through: 0.4, by: 0.1) {
            let f0 = 1 + 0.3 * Double(i)
            for f in stride(from: f0, to: f0 + 0.3, by: 0.05) { steps.append(PathPoint(forward: f, lateral: l, height: 0.15 * Double(i + 1))) }
        } }
        XCTAssertNil(PathObstacleScan.nearest(steps), "stairs are not an obstacle")
        let a = PathObstacleAlerter()
        XCTAssertNil(a.step(t: 0, distanceM: 2.5))
        XCTAssertEqual(a.step(t: 0.2, distanceM: 2.3)?.kind, "obstacle")
        XCTAssertNil(a.step(t: 0.4, distanceM: 2.0))
    }

    func testPhrases() {
        let n = Narrator(n: cfg.narration)
        XCTAssertEqual(n.edge("bollard", 2.4), "Bollard ahead, 2 meters. Watch out.")
        XCTAssertEqual(n.edge("pole", 1.2), "Pole ahead, 1 meter. Watch out.")
        XCTAssertEqual(n.edge("obstacle", 2.6), "Obstacle ahead, 3 meters.")
        XCTAssertEqual(n.edge(CrossingWatch.lefts, 0), "Vehicles approaching from your left.")
        let a = Announcement(t: 0, kind: .edge, trackID: "edge", className: CrossingWatch.right, priority: .nan, reason: "")
        XCTAssertEqual(n.phrase(a, track: nil).1, 60)
        let h = Announcement(t: 0, kind: .threat, trackID: "9", className: "fire hydrant", priority: 1, reason: "threat")
        XCTAssertEqual(n.phrase(h, track: nil).0, "Bollard ahead.")
    }
}
