from ultralytics import YOLO

from rules import Detection


class Detector:
    def __init__(self, model_path, confidence, device, targets, occluders):
        self.model = YOLO(str(model_path))
        if self.model.task != "detect":
            raise ValueError("Model harus mendukung object detection")
        self.names = {int(k): str(v) for k, v in self.model.names.items()}
        available = set(self.names.values())
        unknown = (set(targets) | set(occluders)) - available
        if unknown:
            raise ValueError(f"Kelas tidak ada pada model: {sorted(unknown)}. Tersedia: {sorted(available)}")
        self.allowed_ids = [key for key, name in self.names.items() if name in set(targets) | set(occluders)]
        self.confidence = confidence
        self.device = device

    def detect(self, frame):
        result = self.model.predict(frame, conf=self.confidence, classes=self.allowed_ids,
                                    device=self.device, verbose=False)[0]
        detections = []
        for coords, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(),
                                                 result.boxes.conf.cpu().tolist(),
                                                 result.boxes.cls.cpu().tolist()):
            x1, y1, x2, y2 = map(float, coords)
            if x1 < x2 and y1 < y2:
                index = int(class_id)
                detections.append(Detection(index, self.names[index], float(confidence),
                                            (x1, y1, x2, y2)))
        return detections
