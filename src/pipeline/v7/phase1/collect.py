"""Fail-fast collection of sharded run-index entries and per-case CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.v7.phase1.provenance import (  # noqa: E402
    validate_matched_run_invariants,
    validate_run_index_entry,
)


def collect_indexes(root: Path, output: Path) -> None:
    entries = []
    seen = set()
    for path in sorted(root.rglob("run_index_entry.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        provenance_path = Path(value.get("provenance", ""))
        if not provenance_path.is_absolute():
            provenance_path = path.parent / provenance_path
        if not provenance_path.is_file():
            raise SystemExit(f"missing provenance for {path}: {provenance_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        try:
            validate_run_index_entry(value, provenance)
        except ValueError as exc:
            raise SystemExit(f"invalid run evidence {path}: {exc}") from exc
        video_path = Path(value["video"])
        if not video_path.is_absolute():
            video_path = path.parent / video_path
        if not video_path.is_file():
            raise SystemExit(f"missing generated video for {path}: {video_path}")
        key = (value["case_id"], value["event_id"], int(value["seed"]), value["arm"])
        if key in seen:
            raise SystemExit(f"duplicate run index tuple: {key}")
        seen.add(key)
        entries.append(value)
    if not entries:
        raise SystemExit(f"no run_index_entry.json under {root}")
    try:
        validate_matched_run_invariants(entries, require_complete=True)
    except ValueError as exc:
        raise SystemExit(f"matched four-arm evidence rejected: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def merge_csv(inputs: list[Path], output: Path) -> None:
    rows = []
    fields = None
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input CSV: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fields is None:
                fields = reader.fieldnames
            elif reader.fieldnames != fields:
                raise SystemExit(f"CSV schema mismatch: {path}")
            rows.extend(reader)
    if not rows or not fields:
        raise SystemExit("no CSV rows to merge")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    indexes = sub.add_parser("indexes")
    indexes.add_argument("--root", required=True, type=Path)
    indexes.add_argument("--output", required=True, type=Path)
    csv_parser = sub.add_parser("csv")
    csv_parser.add_argument("--input", action="append", required=True, type=Path)
    csv_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "indexes":
        collect_indexes(args.root, args.output)
    else:
        merge_csv(args.input, args.output)


if __name__ == "__main__":
    main()
