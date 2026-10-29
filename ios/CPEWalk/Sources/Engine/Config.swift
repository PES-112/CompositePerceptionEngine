import Foundation

/// Every constant of the engine, written by prepare_app.py from the Python modules themselves
/// (AppData/cpe_config.json), so the phone and the replay can't drift apart.
struct CPEConfig: Decodable {
    struct Detector: Decodable {
        let imgsz: Int; let conf: Double; let nms_iou: Double; let max_det: Int
        let four_wheelers: [String]; let car_min_width_m: Double; let edge_margin_px: Double
        let legacy_classes: [String]           // answered by CPEDetectorLegacy when it is bundled
    }
    struct Depth: Decodable { let min_m: Double; let max_m: Double; let center_frac: Double; let velocity_window: Int }
    struct GRU: Decodable { let history: Int; let max_objects: Int; let time_bins: Int; let horizon: Int; let step_s: Double }
    struct Noise: Decodable {
        let accel_std, static_accel_std, depth_base, depth_lin, depth_quad, bearing_std_deg, init_speed_std,
            robust_nu, gate_nis: Double
        func depthStd(_ d: Double) -> Double { depth_base + depth_lin * d + depth_quad * d * d }
    }
    struct Fusion: Decodable { let max_gated: Int; let max_misses: Int; let min_hits: Int; let ego_smoothing: Double; let min_walk_speed: Double }
    struct Envelope: Decodable { let ttc_s, corridor_m, envelope_corridor_m, min_closing_ms: Double; let max_coast_frames: Int }
    struct Priority: Decodable { let ignore_threshold, context_in_path, context_near_boundary, context_other_surface: Double }
    struct Announcer: Decodable {
        let cooldown_s, escalation, switch_margin, override_repeat_s, stale_s, clip_s, edge_stale_s, surface_idle_s: Double
    }
    struct Narration: Decodable {
        let fast_closing_ms, near_m, bearing_center_deg, bearing_far_deg: Double
        let distance_buckets_m: [Double]
        let edge_max_m: Int
    }
    struct Segmentation: Decodable {
        let labels: [String: String]
        let walker_rows, walker_cols, edge_cols: [Double]
        let depth_tol_frac, depth_tol_min_m: Double
        let road_logit_bias: Double
        let min_ground_px: Int
        let boundary_share, curb_share, max_edge_m, edge_alert_m: Double
        let edge_confirm_frames: Int
        let edge_clear_s: Double
        let surface_confirm_frames: Int
    }

    let classes: [String]
    let severity: [String: Double]
    let severity_gamma: Double
    let half_width_m: [String: Double]
    let static_classes: [String]
    let detector: Detector
    let depth: Depth
    let gru: GRU
    let noise: Noise
    let fusion: Fusion
    let envelope: Envelope
    let priority: Priority
    let announcer: Announcer
    let narration: Narration
    let segmentation: Segmentation
    let stairs: Stairs

    struct Stairs: Decodable {
        let enter_windows: [[Double]]           // (seconds, minimum rise m)
        let window_s, enter_slope, min_run_m, exit_window_s, exit_rise_m, exit_run_m, warmup_s: Double
        let enter_frames, exit_frames: Int
    }

    func halfWidth(_ cls: String) -> Double { half_width_m[cls] ?? half_width_m["_default"] ?? 0.3 }
    func severityOf(_ cls: String) -> Double { severity[cls] ?? 1.0 }

    static func load(bundle: Bundle = .main) throws -> CPEConfig {
        guard let url = bundle.url(forResource: "cpe_config", withExtension: "json", subdirectory: "AppData")
            ?? bundle.url(forResource: "cpe_config", withExtension: "json") else {
            throw CPEError.missing("AppData/cpe_config.json (run ios/CPEWalk/prepare_app.py)")
        }
        return try JSONDecoder().decode(CPEConfig.self, from: Data(contentsOf: url))
    }
}

enum CPEError: LocalizedError {
    case missing(String)
    var errorDescription: String? {
        switch self { case .missing(let what): return "Missing from the app bundle: \(what)" }
    }
}
