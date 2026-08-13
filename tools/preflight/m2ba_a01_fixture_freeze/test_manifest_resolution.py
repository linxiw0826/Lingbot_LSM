#!/usr/bin/env python3
"""Filesystem-only regression tests for fail-closed manifest resolution."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("fixture_probe", HERE / "probe_fixture_freeze.py")
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ManifestResolutionTest(unittest.TestCase):
    def test_identical_aliases_resolve_by_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "cli", Path(tmp) / "nas"]
            for root in roots:
                root.mkdir()
                (root / "case.json").write_text('{"same": true}\n')
            result = PROBE.resolve_manifest("case", roots)
            self.assertEqual(result["status"], "resolved")
            self.assertEqual(result["selected_path"], str(roots[0] / "case.json"))
            self.assertEqual(len(result["alias_paths"]), 2)
            self.assertEqual(len(result["content_groups"]), 1)

    def test_differing_content_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / "nas", Path(tmp) / "workspace"]
            for index, root in enumerate(roots):
                root.mkdir()
                (root / "case.json").write_text(f'{{"version": {index}}}\n')
            result = PROBE.resolve_manifest("case", roots)
            self.assertEqual(result["status"], "content_conflict")
            self.assertIsNone(result["selected_path"])
            self.assertEqual(len(result["content_groups"]), 2)

    def test_repo_dev_manifest_is_selected_when_only_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_dev = Path(tmp) / "repo" / "src" / "scripts" / "v7" / "dev" / "manifests"
            repo_dev.mkdir(parents=True)
            path = repo_dev / "lookback.json"
            path.write_text('{"event_id": "lookback_non_strict_probe"}\n')
            result = PROBE.resolve_manifest("lookback", [Path(tmp) / "nas", repo_dev])
            self.assertEqual(result["status"], "resolved")
            self.assertEqual(result["selected_path"], str(path))

    def test_empty_reject_records_candidates_without_selecting_memory(self):
        result = PROBE.resolve_memory("EMPTY_REJECT_SAFETY", [64, 80, 96], (64, 112), 176)
        self.assertTrue(result["valid"])
        self.assertEqual(result["candidate_full_frames"], [64, 80, 96])
        self.assertIsNone(result["selected_full_frames"])
        self.assertEqual(result["selection_status"], "not_consumed")

    def test_positive_oracle_selects_complete_ordered_set(self):
        result = PROBE.resolve_memory("TRAIN", [64, 80, 96], (48, 120), 176)
        self.assertTrue(result["valid"])
        self.assertEqual(result["selected_full_frames"], [64, 80, 96])

    def test_positive_oracle_rejects_duplicate_or_out_of_contract_set(self):
        duplicate = PROBE.resolve_memory("TRAIN", [64, 64], (48, 120), 176)
        outside = PROBE.resolve_memory("TRAIN", [40, 80], (48, 120), 176)
        late = PROBE.resolve_memory("TRAIN", [64, 180], (48, 200), 176)
        for result in (duplicate, outside, late):
            self.assertFalse(result["valid"])
            self.assertIsNone(result["selected_full_frames"])


if __name__ == "__main__":
    unittest.main()
