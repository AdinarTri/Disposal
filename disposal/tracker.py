from dataclasses import replace


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0


class Tracker:
    def __init__(self, max_missing_frames, min_iou=0.2):
        self.max_missing_frames = max_missing_frames
        self.min_iou = min_iou
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_index):
        expired = [tid for tid, (_, _, last) in self.tracks.items()
                   if frame_index - last > self.max_missing_frames + 1]
        for tid in expired:
            del self.tracks[tid]
        candidates = sorted(((iou(box, detection.bbox), tid, index)
                             for tid, (box, class_id, _) in self.tracks.items()
                             for index, detection in enumerate(detections)
                             if class_id == detection.class_id), reverse=True)
        matches, used_ids, used_indices = {}, set(), set()
        for score, tid, index in candidates:
            if score < self.min_iou:
                break
            if tid not in used_ids and index not in used_indices:
                matches[index] = tid
                used_ids.add(tid)
                used_indices.add(index)
        result = []
        for index, detection in enumerate(detections):
            tid = matches.get(index)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = (detection.bbox, detection.class_id, frame_index)
            result.append(replace(detection, track_id=tid))
        return result, expired
