import unittest

from shapely.geometry import Polygon

from rules import Detection, Episodes, evaluate
from tracker import Tracker


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.area = Polygon([(10, 10), (90, 10), (90, 90), (10, 90)])

    def decision(self, bbox, others=(), margin=2):
        subject = Detection(0, "Truck", 0.9, bbox)
        return evaluate(subject, [subject, *others], self.area, 100, 100, margin)

    def test_area_and_exceptions(self):
        self.assertEqual(self.decision((12, 12, 20, 20)).status, "VIOLATION")
        self.assertEqual(self.decision((10, 10, 20, 20)).status, "VIOLATION")
        self.assertEqual(self.decision((5, 15, 15, 25)).status, "OUTSIDE")
        self.assertEqual(self.decision((0, 15, 15, 25)).status, "IGNORED_TRUNCATED")
        overlap = Detection(1, "Bus", 0.8, (19, 19, 30, 30))
        touch = Detection(1, "Bus", 0.8, (20, 20, 30, 30))
        self.assertEqual(self.decision((12, 12, 20, 20), [overlap]).status, "IGNORED_OCCLUDED")
        self.assertEqual(self.decision((12, 12, 20, 20), [touch]).status, "VIOLATION")

    def test_one_event_per_episode_and_missing_track(self):
        tracker, episodes = Tracker(2), Episodes()
        inside = Detection(0, "Truck", 0.9, (12, 20, 32, 40))
        outside = Detection(0, "Truck", 0.9, (8, 20, 28, 40))
        events = []
        for frame, detections in enumerate(([inside], [inside], [], [inside], [outside], [inside])):
            tracked, expired = tracker.update(detections, frame)
            episodes.expire(expired)
            for detection in tracked:
                decision = evaluate(detection, tracked, self.area, 100, 100, 2)
                if episodes.update(detection.track_id, decision):
                    events.append(frame)
        self.assertEqual(events, [0, 5])

    def test_exception_holds_episode_until_visible_exit(self):
        episodes = Episodes()
        inside = self.decision((12, 12, 20, 20))
        hidden = self.decision((12, 12, 20, 20), [Detection(1, "Bus", 0.8, (19, 19, 30, 30))])
        outside = self.decision((8, 12, 20, 20), [Detection(1, "Bus", 0.8, (19, 19, 30, 30))])
        self.assertTrue(episodes.update(1, inside))
        self.assertFalse(episodes.update(1, hidden))
        self.assertFalse(episodes.update(1, inside))
        self.assertFalse(episodes.update(1, outside))
        self.assertTrue(episodes.update(1, inside))


if __name__ == "__main__":
    unittest.main()
