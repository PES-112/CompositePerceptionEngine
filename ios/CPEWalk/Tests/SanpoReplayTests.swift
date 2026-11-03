import CoreImage
import simd
import XCTest
@testable import CPEWalk

/// The finished app (WalkPipeline.process: both detectors, tracker, street rules, LiDAR obstacle scan,
/// threat engine, narration, segmentation after the alert) on SANPO test walks prepared by
/// tools/prepare_app_replay.py. Runs only when CPE_REPLAY_DIR is set (xcodebuild: TEST_RUNNER_CPE_REPLAY_DIR);
/// writes <dir>/<session>/app_results.jsonl, one line per processed frame.
///
/// Depth beyond CPE_REPLAY_MAX_DEPTH (default 5 m) is dropped, as the phone's LiDAR has nothing there.
final class SanpoReplayTests: XCTestCase {
    struct Meta: Decodable {
        struct Geo: Decodable { let fx, fy, cx, cy: Double; let sensor_height: Int }
        struct Pose: Decodable { let position, right, forward: [Double] }
        struct Frame: Decodable { let t: Double; let pose: Pose; let rotation_cols: [[Double]]; let position, camera: [Double] }
        let session: String; let image_size, depth_size: [Int]; let focal_px, principal_x: Double
        let depth_geometry: Geo; let frames: [Frame]
    }

    func testReplaySanpoSessions() throws {
        let env = ProcessInfo.processInfo.environment
        guard let dir = env["CPE_REPLAY_DIR"] else { throw XCTSkip("CPE_REPLAY_DIR not set") }
        let root = URL(fileURLWithPath: dir)
        let maxDepth = Float(env["CPE_REPLAY_MAX_DEPTH"] ?? "5") ?? 5
        let only = env["CPE_REPLAY_SESSIONS"].map { Set($0.split(separator: ",").map(String.init)) }
        let sessions = try JSONDecoder().decode([String].self, from: Data(contentsOf: root.appendingPathComponent("sessions.json")))
        for sid in sessions where only?.contains(sid) ?? true {
            let sdir = root.appendingPathComponent(sid)
            guard let data = try? Data(contentsOf: sdir.appendingPathComponent("meta.json")) else {
                print("replay \(sid): not prepared, skipped"); continue
            }
            let meta = try JSONDecoder().decode(Meta.self, from: data)
            try replay(meta, sdir, maxDepth: maxDepth)
        }
    }

    private func replay(_ m: Meta, _ dir: URL, maxDepth: Float) throws {
        let p = WalkPipeline()
        try p.load()
        p.speakAloud = false
        var lines: [String] = []
        p.onFrame = { r in
            var o: [String: Any] = ["t": r.t, "tracks": r.timing.tracks, "det1_ms": r.timing.det1Ms, "det2_ms": r.timing.det2Ms,
                                    "rules_ms": r.timing.rulesMs, "engine_ms": r.timing.engineMs, "alert_ms": r.timing.alertMs,
                                    "seg_ms": r.timing.segMs]
            if let k = r.kind { o["kind"] = k }
            if let c = r.className { o["class"] = c }
            if let s = r.phrase { o["phrase"] = s }
            if let s = r.surface { o["surface"] = s }
            if let h = r.timing.cameraHeightM { o["cam_height_m"] = h }
            if let d = r.timing.obstacleM { o["obstacle_m"] = d }
            let j = try! JSONSerialization.data(withJSONObject: o, options: [.sortedKeys])
            lines.append(String(decoding: j, as: UTF8.self))
        }
        let (dw, dh) = (m.depth_size[0], m.depth_size[1])
        let g = m.depth_geometry
        let started = Date()
        for (k, f) in m.frames.enumerated() {
            let name = String(format: "%04d", k)
            guard let img = CIImage(contentsOf: dir.appendingPathComponent("frames/\(name).jpg")) else { continue }
            let raw = try Data(contentsOf: dir.appendingPathComponent("depth/\(name).f32"))
            var vals = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            for i in vals.indices where !(vals[i].isFinite && vals[i] > 0 && vals[i] <= maxDepth) { vals[i] = 0 }
            let c = f.rotation_cols.map { SIMD3($0[0], $0[1], $0[2]) }
            let geo = DepthGeometry(fx: g.fx, fy: g.fy, cx: g.cx, cy: g.cy, sensorHeight: g.sensor_height,
                                    rotation: simd_double3x3(columns: (c[0], c[1], c[2])),
                                    position: SIMD3(f.position[0], f.position[1], f.position[2]))
            let pose = EgoPose(position: SIMD2(f.pose.position[0], f.pose.position[1]),
                               right: SIMD2(f.pose.right[0], f.pose.right[1]),
                               forward: SIMD2(f.pose.forward[0], f.pose.forward[1]))
            let input = FrameInput(t: f.t, image: img, size: CGSize(width: m.image_size[0], height: m.image_size[1]),
                                   depth: DepthMap(width: dw, height: dh, values: vals), focalPx: m.focal_px,
                                   principalX: m.principal_x, pose: pose,
                                   camera: SIMD3(f.camera[0], f.camera[1], f.camera[2]), depthGeometry: geo)
            autoreleasepool { p.process(input, inputMs: 0) }
        }
        try (lines.joined(separator: "\n") + "\n").write(to: dir.appendingPathComponent("app_results.jsonl"),
                                                       atomically: true, encoding: .utf8)
        let said = lines.filter { $0.contains("\"phrase\"") }.count
        print(String(format: "replay %@: %d frames, %d alerts, %.0f s", m.session, lines.count, said,
                     Date().timeIntervalSince(started)))
    }
}
