"""Track one or more manually calibrated octave regions."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np


class OctaveTracker:
    def __init__(self, gray, quad):
        self.reference = gray
        self.quad = np.asarray(quad, dtype=np.float32)
        self.transform = np.eye(3, dtype=np.float64)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.rint(self.quad).astype(np.int32), 255)
        self.points = cv2.goodFeaturesToTrack(
            gray, maxCorners=48, qualityLevel=0.015, minDistance=7, mask=mask,
        )
        if self.points is None or len(self.points) < 8:
            raise ValueError("Too few features inside the selected octave")

    def update(self, gray):
        guess = cv2.perspectiveTransform(self.points, self.transform)
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(
            self.reference, gray, self.points, guess,
            winSize=(19, 19), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if moved is None:
            return None
        back, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.reference, moved, self.points.copy(),
            winSize=(19, 19), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        )
        if back is None:
            return None
        consistent = ((forward.ravel() == 1) & (backward.ravel() == 1)
                      & (np.linalg.norm((self.points - back).reshape(-1, 2), axis=1) <= 1.3))
        if np.count_nonzero(consistent) < 8:
            return None
        before = self.points[consistent]
        after = moved[consistent]
        affine, inliers = cv2.estimateAffine2D(before, after, ransacReprojThreshold=2)
        if affine is None or inliers.sum() < max(8, len(before) * 0.55):
            return None
        support = before[inliers.ravel() == 1].reshape(-1, 2)
        width, height = np.ptp(self.quad, axis=0)
        if np.ptp(support[:, 0]) < 0.5 * width or np.ptp(support[:, 1]) < 0.25 * height:
            return None
        matrix = np.vstack((affine, [0, 0, 1]))
        quad = cv2.perspectiveTransform(self.quad[None], matrix)[0]
        ratio = cv2.contourArea(quad) / cv2.contourArea(self.quad)
        h, w = gray.shape
        if not (0.5 < ratio < 2 and cv2.isContourConvex(quad)
                and np.all(quad >= 0) and np.all(quad < [w, h])
                and np.linalg.det(affine[:, :2]) > 0):
            return None
        self.transform = matrix
        return quad


def corner_error(quad, target):
    return float(np.linalg.norm(quad - target, axis=1).mean())


def filter_adjacent_regions(quads, anchor, white_counts=None):
    """Reject a neighboring fit if it disagrees with the trusted shared key edge."""
    kept = quads.copy()
    raw_errors = []
    for step in (-1, 1):
        for i in range(anchor + step, len(quads) if step > 0 else -1, step):
            neighbor = kept[i - step]
            quad = kept[i]
            if neighbor is None or quad is None:
                continue
            left, right = (neighbor, quad) if step > 0 else (quad, neighbor)
            errors = np.linalg.norm(left[[1, 2]] - right[[0, 3]], axis=1)
            raw_errors.append(float(errors.mean()))
            white_count = white_counts[i - step] if white_counts is not None else 7
            white_width = np.linalg.norm(neighbor[2] - neighbor[3]) / white_count
            if errors.max() > 0.35 * white_width:
                kept[i] = None
                continue
            aligned = quad.copy()
            if step > 0:
                aligned[[0, 3]] = neighbor[[1, 2]]
            else:
                aligned[[1, 2]] = neighbor[[0, 3]]
            kept[i] = aligned if cv2.isContourConvex(aligned) else None
    return kept, raw_errors


WHITE_OFFSETS = (0, 2, 4, 5, 7, 9, 11)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def keyboard_layout(first, last):
    keys = {}
    white_count = 0
    for midi in range(first, last + 1):
        black = midi % 12 not in WHITE_OFFSETS
        if black:
            left, right, top, bottom = white_count - .27, white_count + .27, .06, .61
        else:
            left, right, top, bottom = white_count + .06, white_count + .94, .73, .98
            white_count += 1
        keys[midi] = (black, [[left, top], [right, top], [right, bottom], [left, bottom]])
    return white_count, keys


def draw_keyboard(frame, quad, layout, color, notes=(), show_black=False, seams=None):
    outline = np.rint(quad).astype(np.int32)
    if seams is not None:
        for back, front in np.rint(seams).astype(np.int32):
            cv2.line(frame, tuple(back), tuple(front), color, 1)
        return
    cv2.polylines(frame, [outline], True, color, 2)
    # Split the calibrated white-key plane in its own coordinates before projecting.
    white_count, keys = layout
    board = np.float32([[0, 0], [white_count, 0], [white_count, 1], [0, 1]])
    projection = cv2.getPerspectiveTransform(board, quad)
    if white_count > 1:
        boundaries = np.float32([[[key, .66 if show_black else 0], [key, 1]]
                                  for key in range(1, white_count)])
        for back, front in np.rint(cv2.perspectiveTransform(boundaries, projection)).astype(np.int32):
            cv2.line(frame, tuple(back), tuple(front), color, 1)
    for midi, (black, corners) in keys.items():
        if midi not in notes and not (show_black and black):
            continue
        polygon = np.rint(cv2.perspectiveTransform(np.float32([corners]), projection)[0]).astype(np.int32)
        if midi in notes:
            cv2.fillConvexPoly(frame, polygon, (45, 215, 255))
            label_x, label_y = polygon.mean(axis=0).astype(int)
            cv2.putText(frame, note_name(midi), (label_x - 9, label_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (15, 20, 30), 1)
        else:
            cv2.polylines(frame, [polygon], True, (110, 180, 245), 1)
    for point in outline:
        cv2.circle(frame, tuple(point), 3, (0, 190, 255), -1)


def run(config, video, output, preview=None, events=None):
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(video))
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or fps <= 0:
            raise ValueError(f"Cannot read {video}")
        start_frame = round(config["start_s"] * fps)
        end_frame = round(config["end_s"] * fps)
        regions = config.get("regions", [config])
        if not regions or end_frame <= start_frame:
            raise ValueError("At least one region and two evaluation frames are required")
        layouts = []
        for region in regions:
            first = region.get("first_midi", 60)
            last = region.get("last_midi", first + 11)
            if ((events is not None and "first_midi" not in region) or
                not 0 <= first <= last <= 127 or first % 12 not in WHITE_OFFSETS or
                last % 12 not in WHITE_OFFSETS):
                raise ValueError("Set the first and last white-key MIDI numbers for each region")
            layouts.append(keyboard_layout(first, last))
        white_counts = [layout[0] for layout in layouts]
        show_black = ["last_midi" in region for region in regions]
        total_keys = sum(len(layout[1]) for layout in layouts)
        seeds = {}
        trackers = []
        for region in regions:
            seed_frame = round(region.get("seed_s", config["start_s"]) * fps)
            if seed_frame > start_frame:
                raise ValueError("Seed frames must be at or before the evaluation start")
            if seed_frame not in seeds:
                capture.set(cv2.CAP_PROP_POS_FRAMES, seed_frame)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode seed frame {seed_frame}")
                seeds[seed_frame] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            started = perf_counter()
            tracker = OctaveTracker(seeds[seed_frame], region["seed_quad_px"])
            trackers.append({
                "id": region.get("id", "octave"),
                "seed_frame": seed_frame,
                "tracker": tracker,
                "references": {round(item["time_s"] * fps): item for item in region.get("checks", [])},
                "seam_references": {round(item["time_s"] * fps): item for item in region.get("seam_checks", [])},
                "seam_samples": [],
                "initialization_ms": (perf_counter() - started) * 1000,
                "samples": [],
                "tracked": 0,
            })
        ids = [item["id"] for item in trackers]
        if len(set(ids)) != len(ids):
            raise ValueError("Region IDs must be unique")
        anchor = ids.index(config["anchor_region"]) if "anchor_region" in config else None
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if preview:
            size = tuple(int(capture.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(preview), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise ValueError("Cannot open preview video")
        timing = []
        frames_by_region_count = [0] * (len(trackers) + 1)
        raw_seam_errors = []
        rejected_regions = 0
        highlighted = set()
        highlighted_frames = 0
        next_event = 0
        active_events = []
        for index in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}")
            start = perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous = [item["tracker"].transform for item in trackers]
            quads = [
                item["tracker"].quad if index == item["seed_frame"] else item["tracker"].update(gray)
                for item in trackers
            ]
            raw_count = sum(quad is not None for quad in quads)
            if anchor is not None:
                quads, seam_errors = filter_adjacent_regions(quads, anchor, white_counts)
                for item, quad, transform in zip(trackers, quads, previous):
                    if quad is None:
                        item["tracker"].transform = transform
            count = sum(quad is not None for quad in quads)
            if index > start_frame:
                timing.append((perf_counter() - start) * 1000)
                frames_by_region_count[count] += 1
                if anchor is not None:
                    raw_seam_errors.extend(seam_errors)
                    rejected_regions += raw_count - count
                for item, quad in zip(trackers, quads):
                    item["tracked"] += int(quad is not None)
            for item, quad, white_count in zip(trackers, quads, white_counts):
                check = item["references"].get(index)
                if check is None:
                    continue
                target = np.asarray(check["quad_px"], dtype=np.float32)
                white_width = np.linalg.norm(target[2] - target[3]) / white_count
                error = corner_error(quad, target) if quad is not None else None
                item["samples"].append({
                    "time_s": check["time_s"],
                    "tracked": quad is not None,
                    "quad_px": quad.tolist() if quad is not None else None,
                    "mean_corner_error_px": error,
                    "error_in_white_key_widths": float(error / white_width) if error is not None else None,
                })
            for region, item, quad in zip(regions, trackers, quads):
                check = item["seam_references"].get(index)
                if check is None:
                    continue
                errors = []
                if quad is not None:
                    transform = cv2.getPerspectiveTransform(item["tracker"].quad, quad)
                    lines = cv2.perspectiveTransform(np.float32(region["white_seams_px"]), transform)
                    for seam, y, x in check["points_yx"]:
                        back, front = lines[seam]
                        predicted = back[0] + (front[0] - back[0]) * (y - back[1]) / (front[1] - back[1])
                        errors.append(float(abs(predicted - x)))
                item["seam_samples"].append({"time_s": check["time_s"], "total": len(check["points_yx"]),
                    "errors_px": errors, "within_5px": sum(error <= 5 for error in errors)})
            if writer:
                played = set()
                if events is not None:
                    timestamp = index / fps
                    while next_event < len(events) and events[next_event]["onset_s"] <= timestamp:
                        active_events.append(events[next_event])
                        next_event += 1
                    active_events = [event for event in active_events if event["end_s"] > timestamp]
                    played = {event["midi"] for event in active_events}
                colors = ((105, 245, 110), (255, 215, 80))
                mapped = set()
                for i, quad in enumerate(quads):
                    if quad is not None:
                        keys = played.intersection(layouts[i][1])
                        seams = regions[i].get("white_seams_px")
                        if seams is not None:
                            reference = trackers[i]["tracker"].quad
                            transform = cv2.getPerspectiveTransform(reference, quad)
                            seams = cv2.perspectiveTransform(np.float32(seams), transform)
                        draw_keyboard(frame, quad, layouts[i], colors[i % len(colors)], keys,
                                      show_black=show_black[i], seams=seams)
                        mapped.update(keys)
                status = f"OCTAVES TRACKED {count}/{len(quads)}"
                if events is not None:
                    highlighted.update(mapped)
                    highlighted_frames += int(bool(mapped))
                    status = "AUDIO: " + (" + ".join(note_name(note) for note in sorted(played)) or "-")
                    if played - mapped:
                        status += "   |   some notes outside tracked keys"
                if any(show_black):
                    status = f"{total_keys} KEYS  |  " + status
                cv2.putText(frame, status,
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .56, (105, 245, 110), 2)
                writer.write(frame)
        if not timing or any(len(item["samples"]) != len(item["references"]) for item in trackers):
            raise ValueError("Evaluation interval or annotated timestamps are invalid")
        result = {
            "video": video.name,
            "start_s": config["start_s"],
            "end_s": config["end_s"],
            "post_seed_frames": len(timing),
            "median_processing_ms": median(timing),
            "p95_processing_ms": float(np.percentile(timing, 95)),
        }
        region_results = [{
            "id": item["id"],
            "initial_points": len(item["tracker"].points),
            "initialization_ms": item["initialization_ms"],
            "accepted_updates": item["tracked"],
            "checks": item["samples"],
            "seam_checks": item["seam_samples"],
        } for item in trackers]
        if "regions" in config:
            result.update(regions=region_results, frames_by_region_count=frames_by_region_count)
            if anchor is not None:
                result["shared_edge_rejections"] = rejected_regions
            if raw_seam_errors:
                result["raw_shared_edge_error_px"] = {
                    "median": median(raw_seam_errors), "p95": float(np.percentile(raw_seam_errors, 95)),
                }
        else:
            result.update({key: value for key, value in region_results[0].items() if key != "id"})
        if events is not None:
            result["audio_overlay"] = {
                "highlighted_frames": highlighted_frames,
                "highlighted_midi": sorted(highlighted),
            }
        if any(show_black):
            result["keyboards"] = [{
                "first_midi": min(keys), "last_midi": max(keys),
                "white_keys": white_count, "black_keys": len(keys) - white_count,
                "keys": [{"midi": midi, "name": note_name(midi), "black": black, "highlight_quad": corners}
                         for midi, (black, corners) in keys.items()],
            } for white_count, keys in layouts]
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        capture.release()
        if writer:
            writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--notes", type=Path, help="Basic Pitch results JSON with onset and end timestamps")
    args = parser.parse_args()
    events = None
    if args.notes:
        if not args.preview:
            parser.error("--notes requires --preview")
        events = json.loads(args.notes.read_text(encoding="utf-8"))["predictions"]
        if any("end_s" not in event for event in events):
            parser.error("Regenerate notes with score_notes.py to include end timestamps")
        events.sort(key=lambda event: event["onset_s"])
    run(json.loads(args.config.read_text(encoding="utf-8")), args.video, args.output, args.preview, events)


if __name__ == "__main__":
    main()
