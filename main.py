import argparse
import json
import math
import sys
from pathlib import Path
from uuid import uuid4

import cv2

from detector import Detector
from reporter import Reporter
from rules import Episodes, evaluate, validate_polygon
from tracker import Tracker
from video_io import draw


def load_config(path, width, height):
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("reference_width", "reference_height"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} harus bilangan bulat positif")
    if (width, height) != (config["reference_width"], config["reference_height"]):
        raise ValueError(f"Resolusi video {width}x{height} berbeda dari konfigurasi "
                         f"{config['reference_width']}x{config['reference_height']}")
    for key in ("main_area", "violation_area"):
        validate_polygon(config[key], width, height, key)
    if not 0 < config["confidence_threshold"] <= 1:
        raise ValueError("confidence_threshold harus > 0 dan <= 1")
    for key in ("target_classes", "occluder_classes"):
        if not isinstance(config.get(key), list) or not config[key] or not all(isinstance(v, str) for v in config[key]):
            raise ValueError(f"{key} harus daftar nama kelas yang tidak kosong")
    if config["frame_edge_margin_px"] < 0 or config["max_missing_frames"] < 0:
        raise ValueError("margin dan max_missing_frames tidak boleh negatif")
    if config["device"] != "cpu" and not str(config["device"]).startswith("cuda"):
        raise ValueError("device harus cpu atau cuda")
    return config


def process(video_path, model_path, config_path, output_dir):
    if not video_path.is_file():
        raise ValueError(f"Video tidak ditemukan: {video_path}")
    if not model_path.is_file():
        raise ValueError(f"Model tidak ditemukan: {model_path}")
    if not config_path.is_file():
        raise ValueError(f"Konfigurasi tidak ditemukan: {config_path}")
    reader = cv2.VideoCapture(str(video_path))
    if not reader.isOpened():
        raise ValueError(f"Video tidak dapat dibaca: {video_path}")
    writer = None
    reporter = None
    try:
        width, height = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = reader.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or fps <= 0 or not math.isfinite(fps):
            raise ValueError("Metadata video tidak valid: lebar, tinggi, atau FPS")
        config = load_config(config_path, width, height)
        polygon = validate_polygon(config["violation_area"], width, height, "violation_area")
        detector = Detector(model_path, config["confidence_threshold"], config["device"],
                            config["target_classes"], config["occluder_classes"])
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Tidak dapat membuka penulis video: {output_dir / 'annotated.mp4'}")
        reporter = Reporter(output_dir, bool(config.get("export_xlsx", False)))
        tracker = Tracker(config["max_missing_frames"])
        episodes = Episodes()
        run_id = uuid4().hex
        target_names = set(config["target_classes"])
        frame_index = 0
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            detections = detector.detect(frame)
            tracked, expired = tracker.update(detections, frame_index)
            episodes.expire(expired)
            labeled = []
            for detection in tracked:
                if detection.class_name not in target_names:
                    continue
                decision = evaluate(detection, tracked, polygon, width, height,
                                    config["frame_edge_margin_px"])
                reporter.counts[decision.status] += 1
                labeled.append((detection, decision))
                if episodes.update(detection.track_id, decision):
                    x1, y1, x2, y2 = detection.bbox
                    reporter.add({"run_id": run_id, "event_id": uuid4().hex,
                                  "video_name": video_path.name, "area_id": config["area_id"],
                                  "track_id": detection.track_id, "class_name": detection.class_name,
                                  "confidence": round(detection.confidence, 6), "frame_index": frame_index,
                                  "video_time_seconds": round(frame_index / fps, 6),
                                  "bbox_x1": round(x1, 3), "bbox_y1": round(y1, 3),
                                  "bbox_x2": round(x2, 3), "bbox_y2": round(y2, 3),
                                  "coverage_ratio": round(decision.coverage_ratio, 6),
                                  "status": decision.status, "reason": decision.reason})
            writer.write(draw(frame, config["main_area"], config["violation_area"],
                              labeled, frame_index / fps))
            reporter.frames += 1
            frame_index += 1
        writer.release()
        writer = None
        if not (output_dir / "annotated.mp4").is_file() or (output_dir / "annotated.mp4").stat().st_size == 0:
            raise OSError("Video hasil gagal ditulis")
        reporter.close("completed")
        reporter = None
        print(f"Selesai: {output_dir} ({frame_index} frame)")
    except BaseException as error:
        if reporter:
            reporter.close("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error)
            reporter = None
        if output_dir.exists():
            print(f"Hasil parsial: {output_dir}", file=sys.stderr)
        raise
    finally:
        reader.release()
        if writer:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description="Deteksi pelanggaran area kendaraan pada video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        process(args.video, args.model, args.config, args.output)
    except (ValueError, OSError, RuntimeError, KeyboardInterrupt, KeyError) as error:
        print(f"Gagal: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
