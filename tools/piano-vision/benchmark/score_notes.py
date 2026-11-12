"""Compare a Basic Pitch audio baseline with independent performance MIDI/TSV."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import Model, predict


def load_notes(path):
    if path.suffix.lower() in (".mid", ".midi"):
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(str(path))
        return [(n.start, n.pitch) for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [(float(row[0]), int(row[3])) for row in csv.reader(stream, delimiter="\t")
                if row and not row[0].startswith("#")]


def match_events(reference, predictions, tolerance):
    truth = defaultdict(list)
    estimates = defaultdict(list)
    for time, pitch in reference:
        truth[pitch].append(time)
    for time, pitch in predictions:
        estimates[pitch].append(time)
    errors = []
    for pitch, actual in truth.items():
        predicted = sorted(estimates[pitch])
        i = j = 0
        actual.sort()
        while i < len(actual) and j < len(predicted):
            delta = predicted[j] - actual[i]
            if delta < -tolerance:
                j += 1
            elif delta > tolerance:
                i += 1
            else:
                errors.append(abs(delta))
                i += 1
                j += 1
    tp = len(errors)
    return {
        "tolerance_ms": round(tolerance * 1000),
        "tp": tp,
        "fp": len(predictions) - tp,
        "fn": len(reference) - tp,
        "precision": tp / len(predictions) if predictions else 0,
        "recall": tp / len(reference) if reference else 0,
        "f1": 2 * tp / (len(reference) + len(predictions)) if reference or predictions else 0,
        "median_matched_onset_error_ms": 1000 * median(errors) if errors else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("reference", help="MIDI/TSV reference path, or - for unverified predictions")
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-shift", type=float, default=0, help="Documented source offset in seconds")
    parser.add_argument("--full-keyboard", action="store_true", help="Allow the standard A0–C8 range")
    args = parser.parse_args()
    reference_path = None if args.reference == "-" else Path(args.reference)
    reference = None
    if reference_path:
        reference = [(time + args.reference_shift, pitch) for time, pitch in load_notes(reference_path)]
        if not reference:
            parser.error("Reference contains no piano notes")
    model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
    _, _, events = predict(
        args.audio, model, onset_threshold=.55, frame_threshold=.35,
        minimum_note_length=90, minimum_frequency=27.5 if args.full_keyboard else 32,
        maximum_frequency=4186.01 if args.full_keyboard else 2100,
        melodia_trick=False,
    )
    predictions = [(float(start), int(pitch)) for start, _, pitch, _, _ in events]
    result = {
        "model": "Spotify Basic Pitch ONNX, audio only",
        "full_keyboard": args.full_keyboard,
        "audio": args.audio.name,
        "reference": reference_path.name if reference_path else None,
        "reference_shift_s": args.reference_shift if reference_path else None,
        "reference_count": len(reference) if reference else None,
        "prediction_count": len(predictions),
        "scores": [match_events(reference, predictions, ms / 1000) for ms in (50, 100)] if reference else [],
        "predictions": [
            {"onset_s": float(start), "end_s": float(end), "midi": int(pitch)}
            for start, end, pitch, _, _ in sorted(events, key=lambda event: (event[0], event[2]))
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}))


if __name__ == "__main__":
    main()
