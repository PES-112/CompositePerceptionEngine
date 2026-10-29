import Foundation

// Port of src/threat_engine/announcer.py: whether to speak this frame, separately from what is most
// dangerous. Lanes, in order: override (preempts), edge ("Curb ahead"), object threat (hysteresis,
// cooldown, stale drop), surface ("Walking on the sidewalk", lowest; cut by any hazard).

enum AnnouncementKind: String { case override, threat, edge, surface }

struct Announcement {
    let t: Double
    let kind: AnnouncementKind
    let trackID: String
    let className: String
    let priority: Double
    let reason: String
    var distanceM: Double? = nil
}

struct EdgeAlert { let t: Double; let kind: String; let distanceM: Double }

final class AnnouncementManager {
    static let surfaceChange = "now on", surfaceReminder = "still on"

    private let c: CPEConfig.Announcer
    private var busyUntil = -1e9
    private var surfacePlaying = false
    private var current: String?
    private var lastSaid: [String: (Double, Double)] = [:]
    private var lastOverride: [String: Double] = [:]
    private var pending: (Double, RankedThreat)?
    private var pendingEdge: (Double, EdgeAlert)?
    private var lastSurface: String?

    init(config: CPEConfig.Announcer) { c = config }

    private func busy(_ t: Double) -> Bool { t < busyUntil && !surfacePlaying }

    func step(t: Double, overrides: [Override], ranked: [RankedThreat], edge: EdgeAlert? = nil,
              surface: String? = nil) -> Announcement? {
        if let edge { pendingEdge = (t, edge) }
        for o in overrides {
            let id = o.track.trackID
            if t - (lastOverride[id] ?? -1e9) >= c.override_repeat_s {
                lastOverride[id] = t
                busyUntil = t + c.clip_s; surfacePlaying = false
                pending = nil
                current = id
                return Announcement(t: t, kind: .override, trackID: id, className: o.track.className,
                                    priority: .infinity, reason: "override")
            }
        }
        if let (seen, e) = pendingEdge {
            if t - seen > c.edge_stale_s {
                pendingEdge = nil
            } else if !busy(t) {
                pendingEdge = nil
                busyUntil = t + c.clip_s; surfacePlaying = false
                return Announcement(t: t, kind: .edge, trackID: "edge:" + e.kind, className: e.kind, priority: .nan,
                                    reason: "edge ahead", distanceM: e.distanceM)
            }
        }
        if let a = threat(t, ranked) { return a }
        if overrides.isEmpty && t >= busyUntil { return surfaceLane(t, surface, quiet: ranked.isEmpty) }
        return nil
    }

    private func threat(_ t: Double, _ ranked: [RankedThreat]) -> Announcement? {
        if let cand = choose(t, ranked), pending == nil || cand.track.trackID != pending!.1.track.trackID {
            pending = (t, cand)
        }
        guard let (chosenAt, th) = pending else { return nil }
        if t - chosenAt > c.stale_s { pending = nil; return nil }
        if busy(t) { return nil }
        pending = nil
        let id = th.track.trackID
        busyUntil = t + c.clip_s; surfacePlaying = false
        current = id
        lastSaid[id] = (t, th.priority)
        return Announcement(t: t, kind: .threat, trackID: id, className: th.track.className, priority: th.priority,
                            reason: "threat")
    }

    private func surfaceLane(_ t: Double, _ surface: String?, quiet: Bool) -> Announcement? {
        guard let surface else { return nil }
        let reason: String
        if surface != lastSurface { reason = Self.surfaceChange }
        else if quiet && t - busyUntil >= c.surface_idle_s { reason = Self.surfaceReminder }
        else { return nil }
        lastSurface = surface
        busyUntil = t + c.clip_s; surfacePlaying = true
        return Announcement(t: t, kind: .surface, trackID: "surface:" + surface, className: surface, priority: .nan,
                            reason: reason)
    }

    private func choose(_ t: Double, _ ranked: [RankedThreat]) -> RankedThreat? {
        guard var top = ranked.first else { return nil }
        if let cur = current, top.track.trackID != cur,
           let held = ranked.first(where: { $0.track.trackID == cur }), top.priority < held.priority * c.switch_margin {
            top = held                                   // hysteresis: stay on the current object
        }
        if let last = lastSaid[top.track.trackID], t - last.0 < c.cooldown_s, top.priority < last.1 * c.escalation {
            return nil
        }
        return top
    }
}
