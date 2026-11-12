"""Measure whole-keyboard tracking before, during and after a controlled mask."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from track_octave import OctaveTracker


def score(video, quad, notes, mask_start, mask_end, rect):
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(video))
    totals = {phase: {"frames": 0, "tracked": 0, "note_frames": 0, "highlightable": 0}
              for phase in ("before", "during", "after")}
    first_recovery = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        ok, seed = capture.read()
        if not ok or fps <= 0:
            raise ValueError(f"Cannot read {video}")
        tracker = OctaveTracker(cv2.cvtColor(seed, cv2.COLOR_BGR2GRAY), quad)
        x, y, width, height = rect
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > seed.shape[1] or y + height > seed.shape[0]:
            raise ValueError("Mask must be inside the source frame")
        start, end = round(mask_start * fps), round(mask_end * fps)
        stop = end + round(2 * fps)
        for index in range(1, stop):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ended at frame {index}")
            if index < start - round(2 * fps):
                tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                continue
            phase = "before" if index < start else "during" if index < end else "after"
            if phase == "during":
                frame[y:y + height, x:x + width] = 0
            result = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            tracked = result is not None
            timestamp = index / fps
            note_active = any(21 <= note["midi"] <= 108 and note["onset_s"] <= timestamp < note["end_s"]
                              for note in notes)
            bucket = totals[phase]
            bucket["frames"] += 1
            bucket["tracked"] += tracked
            bucket["note_frames"] += note_active
            bucket["highlightable"] += tracked and note_active
            if phase == "after" and tracked and first_recovery is None:
                first_recovery = (index - end) * 1000 / fps
    finally:
        capture.release()
    return {
        "video": video.name, "mask_start_s": mask_start, "mask_end_s": mask_end,
        "mask_xywh_px": rect, "frames_per_second": fps,
        "recovery_after_mask_ms": first_recovery, "phases": totals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("notes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask", type=int, nargs=4, default=[70, 300, 1740, 275], metavar=("X", "Y", "W", "H"))
    args = parser.parse_args()
    config = json.loads(args.calibration.read_text(encoding="utf-8"))
    notes = json.loads(args.notes.read_text(encoding="utf-8"))["predictions"]
    if config["start_s"] != 0:
        parser.error("This benchmark requires a frame-zero calibration")
    result = score(args.video, np.float32(config["seed_quad_px"]), notes, 8, 10, args.mask)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
