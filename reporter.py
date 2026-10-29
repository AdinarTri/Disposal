import csv
import json
from collections import Counter
from time import monotonic


FIELDS = ["run_id", "event_id", "video_name", "area_id", "track_id", "class_name",
          "confidence", "frame_index", "video_time_seconds", "bbox_x1", "bbox_y1",
          "bbox_x2", "bbox_y2", "coverage_ratio", "status", "reason"]


class Reporter:
    def __init__(self, output_dir, export_xlsx):
        self.output_dir = output_dir
        self.export_xlsx = export_xlsx
        self.file = (output_dir / "violations.csv").open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.file.flush()
        self.counts = Counter()
        self.events = 0
        self.frames = 0
        self.started = monotonic()

    def add(self, row):
        self.writer.writerow(row)
        self.file.flush()
        self.events += 1

    def close(self, status, error=None):
        self.file.close()
        duration = monotonic() - self.started
        summary = {"status": status, "frames_processed": self.frames, "event_count": self.events,
                   "evaluations_by_status": dict(self.counts), "processing_seconds": round(duration, 3),
                   "processing_fps": round(self.frames / duration, 3) if duration else 0}
        if error:
            summary["error"] = str(error)
        if self.export_xlsx and status == "completed":
            from openpyxl import Workbook
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Violations"
            with (self.output_dir / "violations.csv").open(newline="", encoding="utf-8-sig") as source:
                for row in csv.reader(source):
                    sheet.append(row)
            workbook.save(self.output_dir / "violations.xlsx")
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
