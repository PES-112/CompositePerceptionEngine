import unittest

from src.narration.templates import narrate
from src.segmentation.edge_alert import EdgeAlert, SurfaceTracker
from src.segmentation.labels import ROAD, WALKABLE
from src.threat_engine import envelope
from src.threat_engine.announcer import AnnouncementManager
from src.threat_engine.engine import ThreatEngine
from tests.test_threat_engine import ranked_threat, track


def curb(t=0.0, d=3.2):
    return EdgeAlert(t, "curb", d)


class EdgeAnnouncementTests(unittest.TestCase):
    def test_edge_goes_before_an_object_threat(self):
        mgr = AnnouncementManager()
        a = mgr.step(0.0, [], [ranked_threat("a", 0.5)], curb())
        self.assertEqual((a.kind, a.class_name, a.distance_m), ("edge", "curb", 3.2))
        self.assertIsNone(mgr.step(0.2, [], [ranked_threat("a", 0.5)]))        # clip still playing
        self.assertIsNone(mgr.step(1.4, [], [ranked_threat("a", 0.5)]))        # threat chosen at 0.2 went stale
        self.assertEqual(mgr.step(1.6, [], [ranked_threat("a", 0.5)]).kind, "threat")

    def test_override_preempts_an_edge(self):
        mgr = AnnouncementManager()
        o = envelope.check(track(track_id="b", distance_m=1.0, closing_speed_ms=2.0))
        self.assertEqual(mgr.step(0.0, [o], [], curb()).kind, "override")
        self.assertIsNone(mgr.step(0.4, [], []))                                # override playing, edge pending
        self.assertEqual(mgr.step(1.2, [], []).kind, "edge")                    # free again within EDGE_STALE_S

    def test_edge_waits_for_a_playing_clip_then_goes_stale(self):
        mgr = AnnouncementManager(clip_s=3.0)
        mgr.step(0.0, [], [ranked_threat("a", 0.5)])                           # plays until 3.0
        self.assertIsNone(mgr.step(0.2, [], [], curb(0.2)))
        self.assertIsNone(mgr.step(1.6, [], []))                                # 1.4 s > EDGE_STALE_S: dropped
        self.assertIsNone(mgr.step(3.2, [], []))

    def test_engine_turns_an_edge_into_a_narrator_event(self):
        engine = ThreatEngine()
        frame = engine.step(0, 0.0, [], edge=curb())
        event = frame.narrator_event()
        self.assertEqual((event.object_class, event.distance_m, event.reason, event.is_override),
                         ("curb", 3.2, "edge ahead", False))



class SurfaceAnnouncementTests(unittest.TestCase):
    def test_surface_said_at_start_then_every_idle_period(self):
        mgr = AnnouncementManager()
        a = mgr.step(0.0, [], [], surface="walkable")
        self.assertEqual((a.kind, a.reason), ("surface", "now on"))
        self.assertIsNone(mgr.step(3.0, [], [], surface="walkable"))           # 1.2 + 5 s not yet up
        a = mgr.step(6.2, [], [], surface="walkable")
        self.assertEqual((a.kind, a.reason), ("surface", "still on"))

    def test_change_is_said_at_once(self):
        mgr = AnnouncementManager()
        mgr.step(0.0, [], [], surface="walkable")
        self.assertIsNone(mgr.step(0.6, [], [], surface="road"))                # clip still playing
        self.assertEqual(mgr.step(1.4, [], [], surface="road").reason, "now on")

    def test_no_reminder_while_a_threat_is_in_view(self):
        mgr = AnnouncementManager(cooldown_s=100.0)
        mgr.step(0.0, [], [], surface="walkable")
        self.assertEqual(mgr.step(2.0, [], [ranked_threat("a", 0.5)], surface="walkable").kind, "threat")
        self.assertIsNone(mgr.step(9.0, [], [ranked_threat("a", 0.5)], surface="walkable"))   # in cooldown, not quiet
        self.assertEqual(mgr.step(9.2, [], [], surface="walkable").reason, "still on")

    def test_a_hazard_cuts_a_surface_clip(self):
        mgr = AnnouncementManager()
        mgr.step(0.0, [], [], surface="walkable")
        self.assertEqual(mgr.step(0.2, [], [ranked_threat("a", 0.5)]).kind, "threat")
        mgr = AnnouncementManager()
        mgr.step(0.0, [], [], surface="walkable")
        self.assertEqual(mgr.step(0.2, [], [], curb(0.2)).kind, "edge")

    def test_tracker_ignores_flicker(self):
        tr = SurfaceTracker(confirm_frames=3)
        self.assertEqual([tr.step(c) for c in (WALKABLE,) * 3], [None, None, "walkable"])
        self.assertEqual([tr.step(c) for c in (ROAD, WALKABLE, ROAD, None, ROAD, ROAD)],
                         ["walkable"] * 5 + ["road"])

    def test_engine_narrates_ground(self):
        engine = ThreatEngine()
        self.assertEqual(narrate(engine.step(0, 0.0, [], surface="road").narrator_event()), "Now on the road. Careful.")
        self.assertEqual(narrate(engine.step(1, 1.4, [], edge=curb(1.4)).narrator_event()), "Curb ahead, 3 meters.")


if __name__ == "__main__":
    unittest.main()
