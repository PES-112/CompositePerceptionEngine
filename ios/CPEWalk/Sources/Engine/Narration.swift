import Foundation

// Port of src/narration/templates.py: a short deterministic phrase per announcement.

struct Narrator {
    let n: CPEConfig.Narration

    static let edgeWords = ["curb": "Curb", "stairs": "Stairs", "road": "Road", "obstacle": "Obstacle"]
    // Phone-only street rules (StreetRules.swift)
    static let postWords = ["bollard": "Bollard", "pole": "Pole"]
    static let crossingPhrases = [CrossingWatch.left: ("Vehicle approaching from your left.", -60.0),
                                  CrossingWatch.lefts: ("Vehicles approaching from your left.", -60.0),
                                  CrossingWatch.right: ("Vehicle approaching from your right.", 60.0),
                                  CrossingWatch.rights: ("Vehicles approaching from your right.", 60.0),
                                  CrossingWatch.both: ("Vehicles approaching from both sides.", 0.0)]
    /// Spoken class names that differ from the detector's: on Indian streets the "fire hydrant" the
    /// detector reports is almost always a bollard (2026-11-02 walks).
    static let spokenNames = ["fire hydrant": "bollard"]
    static let stairPhrases = ["stairs_up": "Going up stairs.", "stairs_down": "Going down stairs. Careful.",
                               "stairs_end": "End of stairs."]
    static let surfaceWords = ["walkable": "the sidewalk", "crosswalk": "the crosswalk", "road": "the road",
                               "curb": "the curb", "stairs": "stairs", "non-walkable ground": "rough ground"]
    static let unsafeSurfaces: Set<String> = ["road", "curb", "stairs", "non-walkable ground"]

    func direction(_ bearing: Double) -> String {
        let mag = abs(bearing), side = bearing < 0 ? "left" : "right"
        if mag <= n.bearing_center_deg { return "ahead" }
        if mag >= n.bearing_far_deg { return "far \(side)" }
        return side
    }

    func distanceWords(_ d: Double) -> String {
        for b in n.distance_buckets_m where d <= b { return "\(Int(b)) meters" }
        return "far"
    }

    func edge(_ kind: String, _ d: Double) -> String {
        if let p = Self.stairPhrases[kind] { return p }
        if let p = Self.crossingPhrases[kind] { return p.0 }
        let m = Int(min(Double(n.edge_max_m), max(1, d.rounded())))
        if let w = Self.postWords[kind] { return "\(w) ahead, \(m) meter\(m > 1 ? "s" : ""). Watch out." }
        return "\(Self.edgeWords[kind] ?? "Edge") ahead, \(m) meter\(m > 1 ? "s" : "")."
    }

    func surface(_ s: String, changed: Bool) -> String {
        let w = Self.surfaceWords[s] ?? s
        if !changed { return "Walking on \(w)." }
        return "Now on \(w)." + (Self.unsafeSurfaces.contains(s) ? " Careful." : "")
    }

    /// The phrase and the bearing to pan it to (0 = centre).
    func phrase(_ a: Announcement, track: FusedTrack?) -> (String, Double) {
        switch a.kind {
        case .edge: return (edge(a.className, a.distanceM ?? 0), Self.crossingPhrases[a.className]?.1 ?? 0)
        case .surface: return (surface(a.className, changed: a.reason == AnnouncementManager.surfaceChange), 0)
        case .override, .threat:
            let name = Self.spokenNames[a.className] ?? a.className
            let cls = name.prefix(1).uppercased() + name.dropFirst()
            guard let t = track else { return ("\(cls) ahead.", 0) }
            let d = direction(t.bearingDeg)
            if a.kind == .override { return ("Stop! \(cls) very close \(d).", t.bearingDeg) }
            if max(0, t.closingSpeedMs) >= n.fast_closing_ms { return ("\(cls) fast from your \(d).", t.bearingDeg) }
            if t.distanceM <= n.near_m { return ("\(cls) close, \(d).", t.bearingDeg) }
            return ("\(cls) \(d), \(distanceWords(t.distanceM)).", t.bearingDeg)
        }
    }
}
