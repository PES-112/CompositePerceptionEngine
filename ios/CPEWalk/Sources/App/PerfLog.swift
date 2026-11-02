import Foundation
import UIKit

/// Per-frame timing of the live app, one CSV per walk in the app's Documents folder (visible in the
/// Files app under On My iPhone › CPE Walk, or pulled with `xcrun devicectl device copy from`).
/// The columns follow ios/CPETiming's CSV, so `ios/CPETiming/analyze_timing.py` summarises both:
///
///   block      warmup (the first `warmup` frames, excluded from the summary) or walk
///   frame, t_s processed-frame index and seconds since Start (ARKit clock)
///   input_ms   ARFrame -> FrameInput: image rotation, LiDAR depth copy (ARKit queue)
///   det_ms     all detectors (det1_ms the IDD model, det2_ms the legacy v3 model; 0 if absent)
///   track_ms   rest of Stage 1: filtering, tracker, depth per box, closing speed
///   seg_ms     segmentation + scene context (0 on frames it skips)
///   engine_ms  fusion, envelope, GRU, priority, announcer, narration
///   total_ms   input_ms + everything on the pipeline queue
///   thermal_state  ProcessInfo.ThermalState (0 nominal … 3 critical)
///   battery, battery_state  UIDevice level and state (1 unplugged, 2 charging, 3 full)
final class PerfLog {
    struct Row {
        var inputMs = 0.0, det1Ms = 0.0, det2Ms = 0.0, trackMs = 0.0, segMs = 0.0, engineMs = 0.0, totalMs = 0.0
        var tracks = 0
    }

    let url: URL
    private let handle: FileHandle
    private let warmup = 10
    private var frame = 0
    private var pending = ""
    private let lock = NSLock()
    private var battery: (level: Float, state: Int) = (-1, 0)     // read on the main thread (UIDevice)

    init(detectors: Int, segmenter: Bool) throws {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        url = docs.appendingPathComponent("cpewalk_perf_\(stamp).csv")
        var sys = utsname(); uname(&sys)
        let model = withUnsafeBytes(of: &sys.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let header = "# device=\(model) os=\(ProcessInfo.processInfo.operatingSystemVersionString) app=CPEWalk "
            + "detectors=\(detectors) segmenter=\(segmenter ? "yes" : "no") rate=5Hz mode=walk\n"
            + "block,frame,t_s,input_ms,det_ms,det1_ms,det2_ms,track_ms,seg_ms,engine_ms,total_ms,tracks,"
            + "thermal_state,battery,battery_state\n"
        FileManager.default.createFile(atPath: url.path, contents: Data(header.utf8))
        handle = try FileHandle(forWritingTo: url)
        handle.seekToEndOfFile()
        DispatchQueue.main.async { UIDevice.current.isBatteryMonitoringEnabled = true }
        refreshBattery()
    }

    /// Called on the pipeline queue, once per processed frame.
    func add(t: Double, _ r: Row) {
        lock.lock(); let bat = battery; lock.unlock()
        let f = { (x: Double) in String(format: "%.3f", x) }
        pending += [frame < warmup ? "warmup" : "walk", String(frame), f(t), f(r.inputMs), f(r.det1Ms + r.det2Ms),
                    f(r.det1Ms), f(r.det2Ms), f(r.trackMs), f(r.segMs), f(r.engineMs), f(r.totalMs), String(r.tracks),
                    String(ProcessInfo.processInfo.thermalState.rawValue), String(format: "%.2f", bat.level),
                    String(bat.state)].joined(separator: ",") + "\n"
        frame += 1
        if frame % 25 == 0 { flush(); refreshBattery() }     // every 5 s
    }

    func close() {
        flush()
        try? handle.close()
    }

    private func refreshBattery() {
        DispatchQueue.main.async { [weak self] in
            let d = UIDevice.current, b = (d.batteryLevel, d.batteryState.rawValue)
            self?.lock.lock(); self?.battery = b; self?.lock.unlock()
        }
    }

    private func flush() {
        guard !pending.isEmpty else { return }
        handle.write(Data(pending.utf8))
        pending = ""
    }
}

/// Milliseconds since `start` (from `DispatchTime.now().uptimeNanoseconds`).
func msSince(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
