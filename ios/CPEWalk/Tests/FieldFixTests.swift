import XCTest
@testable import CPEWalk

/// Mirrors tests/test_field_fixes.py: stairs from height, the two-wheeler width check, stair phrases.
final class FieldFixTests: XCTestCase {
    private var cfg: CPEConfig!

    override func setUpWithError() throws { cfg = try CPEConfig.load(bundle: Bundle(for: WalkPipeline.self)) }

    /// 5 Hz poses walking along x at `speed` m/s, rising `climb` m/s.
    private func walk(_ s: StairMotion, _ t: inout Double, _ x: inout Double, _ h: inout Double,
                      seconds: Double, speed: Double, climb: Double) -> [(Double, String)] {
        var out: [(Double, String)] = []
        for _ in 0..<Int(seconds * 5) {
            t += 0.2; x += speed * 0.2; h += climb * 0.2
            if let e = s.step(t: t, x: x, y: 0, h: h) { out.append((t, e)) }
        }
        return out
    }

    func testUpAFlightThenEnd() {
        let s = StairMotion(cfg.stairs)
        var t = 0.0, x = 0.0, h = 0.0
        XCTAssertTrue(walk(s, &t, &x, &h, seconds: 3, speed: 1.2, climb: 0).isEmpty)
        let up = walk(s, &t, &x, &h, seconds: 5, speed: 0.5, climb: 0.3)
        XCTAssertEqual(up.map(\.1), [StairMotion.up])
        XCTAssertLessThan(up[0].0 - 3, 1.5)
        XCTAssertEqual(walk(s, &t, &x, &h, seconds: 3, speed: 1.2, climb: 0).map(\.1), [StairMotion.end])
    }

    func testSlowDescentAndFlatAndRamp() {
        var s = StairMotion(cfg.stairs)
        var t = 0.0, x = 0.0, h = 0.0
        _ = walk(s, &t, &x, &h, seconds: 3, speed: 1.0, climb: 0)
        XCTAssertEqual(walk(s, &t, &x, &h, seconds: 6, speed: 0.28, climb: -0.17).map(\.1), [StairMotion.down])
        s = StairMotion(cfg.stairs); t = 0; x = 0; h = 0
        XCTAssertTrue(walk(s, &t, &x, &h, seconds: 30, speed: 1.3, climb: 0).isEmpty)
        XCTAssertTrue(walk(s, &t, &x, &h, seconds: 20, speed: 1.2, climb: 0.1).isEmpty)   // 1:12 ramp
    }

    func testVehicleWidth() {
        let c = cfg.detector, fx = 1000.0, w = 1440.0
        func box(_ x1: Double, _ x2: Double) -> CGRect { CGRect(x: x1, y: 500, width: x2 - x1, height: 200) }
        XCTAssertEqual(Stage1.vehicleClass("car", box(600, 720), 6, fx, w, c), "motorcycle")
        XCTAssertEqual(Stage1.vehicleClass("car", box(500, 800), 6, fx, w, c), "car")
        XCTAssertEqual(Stage1.vehicleClass("car", box(0, 100), 6, fx, w, c), "car")
        XCTAssertEqual(Stage1.vehicleClass("car", box(600, 720), nil, fx, w, c), "car")
        XCTAssertEqual(Stage1.vehicleClass("person", box(600, 640), 6, fx, w, c), "person")
    }

    func testStairPhrases() {
        let n = Narrator(n: cfg.narration)
        XCTAssertEqual(n.edge(StairMotion.up, 0), "Going up stairs.")
        XCTAssertEqual(n.edge(StairMotion.down, 0), "Going down stairs. Careful.")
        XCTAssertEqual(n.edge(StairMotion.end, 0), "End of stairs.")
    }
}
