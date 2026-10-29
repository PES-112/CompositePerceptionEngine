import XCTest
@testable import CPEWalk

/// The Swift engine against the Python one on a recorded SANPO test walk (AppData/golden_session.json,
/// written by prepare_app.py): same Stage-1 rows and camera poses in, same fused tracks, GRU inputs,
/// P(arrive), ranking and spoken phrases out. P(arrive) is compared loosely: the phone runs the
/// FP16 Core ML export (max |Δp| ≈ 1e-3 against PyTorch on random inputs).
final class GoldenSessionTests: XCTestCase {
    struct Golden: Decodable {
        struct Pose: Decodable { let position: [Double]; let right: [Double]; let forward: [Double] }
        struct Row: Decodable {
            let track_id: String; let `class`: String; let confidence: Double; let distance_m: Double?
            let bearing_deg: Double; let velocity_ms: Double; let bbox_area_px: Double
        }
        struct Track: Decodable {
            let id: String; let lateral, forward, closing, lateral_speed, position_std: Double
            let confirmed, measured: Bool
        }
        struct Ann: Decodable { let kind: String; let track: String; let text: String }
        struct Frame: Decodable {
            let k: Int; let t: Double; let pose: Pose; let rows: [Row]; let tracks: [Track]
            let gru_x: [String: [[Double]]]; let p_arrive: [String: Double]; let ranked: [[AnyCodableNumberOrString]]
            let overrides: [String]; let announcement: Ann?
        }
        let session: String
        let frames: [Frame]
    }

    /// ranked rows are [track_id, priority].
    enum AnyCodableNumberOrString: Decodable {
        case s(String), d(Double)
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { self = .s(s) } else { self = .d(try c.decode(Double.self)) }
        }
        var string: String? { if case .s(let v) = self { return v }; return nil }
    }

    func testEngineMatchesPython() throws {
        let bundle = Bundle(for: WalkPipeline.self)
        let url = try XCTUnwrap(bundle.url(forResource: "golden_session", withExtension: "json", subdirectory: "AppData"))
        let g = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
        let cfg = try CPEConfig.load(bundle: bundle)
        let engine = ThreatEngine(config: cfg, predictor: try GRUArrivalPredictor(config: cfg, bundle: bundle))
        let narrator = Narrator(n: cfg.narration)

        var worstTrack = ""
        var maxTrack = 0.0, maxX = 0.0, maxP = 0.0, rankMismatch = 0, phraseMatch = 0, phraseTotal = 0
        for f in g.frames {
            let pose = EgoPose(position: SIMD2(f.pose.position[0], f.pose.position[1]),
                               right: SIMD2(f.pose.right[0], f.pose.right[1]),
                               forward: SIMD2(f.pose.forward[0], f.pose.forward[1]))
            let rows = f.rows.map { PerceptionRow(trackID: $0.track_id, className: $0.class, confidence: $0.confidence,
                                                  distanceM: $0.distance_m, bearingDeg: $0.bearing_deg,
                                                  velocityMs: $0.velocity_ms, bboxAreaPx: $0.bbox_area_px) }
            let out = try engine.step(k: f.k, t: f.t, rows: rows, pose: pose)

            XCTAssertEqual(Set(out.tracks.map(\.trackID)), Set(f.tracks.map(\.id)), "frame \(f.k): track set")
            for gt in f.tracks {
                guard let st = out.tracks.first(where: { $0.trackID == gt.id }) else { continue }
                for (name, a, b) in [("lateral", st.lateralM, gt.lateral), ("forward", st.forwardM, gt.forward),
                                     ("closing", st.closingSpeedMs, gt.closing),
                                     ("lateral_speed", st.lateralSpeedMs, gt.lateral_speed),
                                     ("position_std", st.positionStdM, gt.position_std)] where abs(a - b) > maxTrack {
                    maxTrack = abs(a - b)
                    worstTrack = "frame \(f.k) track \(gt.id) \(name): swift \(a) python \(b)"
                }
                XCTAssertEqual(st.confirmed, gt.confirmed)
                XCTAssertEqual(st.measured, gt.measured)
            }
            for (id, gx) in f.gru_x {
                let sx = engine.predictor.features.features(id, frame: f.k)
                for (i, row) in gx.enumerated() { for (j, v) in row.enumerated() { maxX = max(maxX, abs(Double(sx[i][j]) - v)) } }
            }
            for (id, p) in f.p_arrive { if let sp = out.arrivals[id]?.pArrive { maxP = max(maxP, abs(sp - p)) } }
            let pyRanked = f.ranked.compactMap { $0.first?.string }
            if pyRanked != out.ranked.map(\.track.trackID) { rankMismatch += 1 }
            if let a = f.announcement {
                phraseTotal += 1
                if let sa = out.announcement {
                    let (text, _) = narrator.phrase(sa, track: out.tracks.first { $0.trackID == sa.trackID })
                    if text == a.text { phraseMatch += 1 }
                }
            }
        }
        print("golden \(g.session): max |Δ track| \(maxTrack) (\(worstTrack)), max |Δ x| \(maxX), max |Δ p| \(maxP), " +
              "ranking mismatches \(rankMismatch)/\(g.frames.count), phrases \(phraseMatch)/\(phraseTotal)")
        XCTAssertLessThan(maxTrack, 1e-6, "Kalman fusion drifted from fusion.py")
        XCTAssertLessThan(maxX, 1e-5, "GRU features drifted from samples.history_features")
        XCTAssertLessThan(maxP, 0.02, "GRU probabilities (FP16 Core ML vs PyTorch)")
        XCTAssertLessThanOrEqual(Double(rankMismatch), 0.05 * Double(g.frames.count), "ranking order")
        XCTAssertGreaterThanOrEqual(Double(phraseMatch), 0.9 * Double(phraseTotal), "announcements and phrases")
    }

    func testVelocityMatchesPhysics() {
        // physics.compute_velocity: least squares over (t, d), positive approaching, clamped at 0
        XCTAssertEqual(Stage1.velocity([(0, 10), (0.2, 9.8), (0.4, 9.6)]), 1.0, accuracy: 1e-9)
        XCTAssertEqual(Stage1.velocity([(0, 5), (0.2, 5.4)]), 0)
    }

    func testNarrationPhrases() throws {
        let cfg = try CPEConfig.load(bundle: Bundle(for: WalkPipeline.self))
        let n = Narrator(n: cfg.narration)
        XCTAssertEqual(n.edge("curb", 3.2), "Curb ahead, 3 meters.")
        XCTAssertEqual(n.edge("stairs", 0.6), "Stairs ahead, 1 meter.")
        XCTAssertEqual(n.surface("road", changed: true), "Now on the road. Careful.")
        XCTAssertEqual(n.surface("walkable", changed: false), "Walking on the sidewalk.")
        XCTAssertEqual(n.direction(-50), "far left")
        XCTAssertEqual(Speech.pan(-120), -1)
    }
}
