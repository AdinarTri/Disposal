from dataclasses import dataclass

from shapely.geometry import Polygon, box


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    track_id: int = -1


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    coverage_ratio: float
    truncated: bool
    occluded: bool


def validate_polygon(points, width, height, name):
    if not isinstance(points, list) or len(points) < 3 or any(
        not isinstance(p, list) or len(p) != 2 or
        any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in p)
        for p in points
    ):
        raise ValueError(f"{name}: perlu minimal tiga titik [x, y]")
    if len({tuple(p) for p in points}) != len(points):
        raise ValueError(f"{name}: titik harus berbeda")
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        raise ValueError(f"{name}: titik harus berada di dalam frame {width}x{height}")
    polygon = Polygon(points)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError(f"{name}: polygon berpotongan sendiri atau luas nol")
    return polygon


def evaluate(detection, detections, polygon, width, height, margin):
    x1, y1, x2, y2 = detection.bbox
    if x1 >= x2 or y1 >= y2:
        return Decision("INVALID", "INVALID_BBOX", 0, False, False)
    shape = box(x1, y1, x2, y2)
    coverage = min(1.0, max(0.0, polygon.intersection(shape).area / shape.area))
    inside = polygon.covers(shape)
    truncated = x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin
    occluded = any(
        other is not detection and
        max(x1, other.bbox[0]) < min(x2, other.bbox[2]) and
        max(y1, other.bbox[1]) < min(y2, other.bbox[3])
        for other in detections
    )
    if truncated:
        return Decision("IGNORED_TRUNCATED", "FRAME_EDGE", coverage, True, occluded)
    if occluded:
        return Decision("IGNORED_OCCLUDED", "VEHICLE_OVERLAP", coverage, False, True)
    if not inside:
        return Decision("OUTSIDE", "BBOX_NOT_COVERED", coverage, False, False)
    return Decision("VIOLATION", "FULL_BBOX_INSIDE", coverage, False, False)


class Episodes:
    def __init__(self):
        self.active = set()

    def update(self, track_id, decision):
        if decision.status == "OUTSIDE":
            self.active.discard(track_id)
        elif decision.status.startswith("IGNORED") and decision.coverage_ratio < 1:
            self.active.discard(track_id)
        elif decision.status == "VIOLATION" and track_id not in self.active:
            self.active.add(track_id)
            return True
        return False

    def expire(self, track_ids):
        self.active.difference_update(track_ids)
