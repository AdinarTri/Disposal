import cv2
import numpy as np


def draw(frame, main_points, violation_points, labeled, time_seconds):
    cv2.polylines(frame, [np.asarray(main_points, dtype=np.int32)], True, (255, 180, 0), 2)
    cv2.polylines(frame, [np.asarray(violation_points, dtype=np.int32)], True, (0, 0, 255), 2)
    cv2.putText(frame, f"video {time_seconds:.2f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    colors = {"VIOLATION": (0, 0, 255), "OUTSIDE": (0, 180, 0),
              "IGNORED_TRUNCATED": (0, 180, 255), "IGNORED_OCCLUDED": (0, 180, 255)}
    for detection, decision in labeled:
        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        color = colors.get(decision.status, (150, 150, 150))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (f"#{detection.track_id} {detection.class_name} {detection.confidence:.2f} "
                 f"{decision.status} {decision.reason}")
        text_y = max(18, y1 - 7)
        cv2.putText(frame, label, (max(0, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)
    return frame
