import Foundation

/// Port of src/threat_engine/stairs.py StairMotion: "Going up / down stairs." and "End of stairs." from
/// the walker's own height (ARKit world tracking, gravity-aligned), which the segmenter cannot give
/// once the walker is on the flight.
final class StairMotion {
    static let up = "stairs_up", down = "stairs_down", end = "stairs_end"

    private let c: CPEConfig.Stairs
    private var hist: [(t: Double, x: Double, y: Double, h: Double)] = []
    private var t0: Double?
    private(set) var direction: String?
    private var enterRun = 0, enterDir: String?, exitRun = 0

    init(_ c: CPEConfig.Stairs) { self.c = c }

    var onStairs: Bool { direction != nil }

    /// x, y: ground-plane position (m); h: height (m, up positive). Returns an event kind on a change.
    func step(t: Double, x: Double, y: Double, h: Double) -> String? {
        if t0 == nil { t0 = t }
        hist.append((t, x, y, h))
        while hist.count > 1 && t - hist[1].t >= c.window_s { hist.removeFirst() }
        if t - t0! < c.warmup_s || t - hist[0].t < c.window_s - 1e-6 { return nil }

        if direction == nil {
            var d: String?
            for w in c.enter_windows {
                let s = since(t, w[0])
                let rise = h - s.h, run = hypot(x - s.x, y - s.y)
                if abs(rise) >= w[1] && abs(rise) >= c.enter_slope * run && run >= c.min_run_m {
                    d = rise > 0 ? Self.up : Self.down
                    break
                }
            }
            enterRun = d != nil && d == enterDir ? enterRun + 1 : (d != nil ? 1 : 0)
            enterDir = d
            if enterRun >= c.enter_frames, let d {
                direction = d; enterRun = 0; exitRun = 0
                return d
            }
            return nil
        }
        let s = since(t, c.exit_window_s)
        let flat = abs(h - s.h) < c.exit_rise_m && hypot(x - s.x, y - s.y) >= c.exit_run_m
        exitRun = flat ? exitRun + 1 : 0
        if exitRun >= c.exit_frames {
            direction = nil; exitRun = 0; enterDir = nil
            return Self.end
        }
        return nil
    }

    /// The oldest kept pose no more than `window` s old.
    private func since(_ t: Double, _ window: Double) -> (t: Double, x: Double, y: Double, h: Double) {
        hist.first { t - $0.t <= window + 1e-6 }!
    }
}
