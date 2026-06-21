"""Unified output / log path conventions for the Lingbot_LSM pipeline.

This module centralizes *where* training and evaluation artifacts are written so
that Phase V (v5) scripts share a single, predictable layout. It is intentionally
pure-stdlib (os / pathlib / datetime, with an optional PyYAML import for config
snapshots) and pulls in **no** heavy deps (torch / wan / etc.), so it is cheap to
import from anywhere.

Directory convention
--------------------
All artifacts live under a single root (overridable via the ``OUTPUT_ROOT`` env
var; default ``/home/nvme02/wlx/Memory/outputs``)::

    OUTPUT_ROOT/
      <version>/                       # e.g. "v5"
        train/
          <run_name>/                  # e.g. "v5_base_20260621_101500"
            checkpoints/               # model weights
            logs/                      # co-located training logs
            samples/                   # periodic sample dumps
            config.yaml                # snapshot of the config this run used
        eval/
          INDEX.md                     # running ledger of all eval verdicts
          <run_name>/
            <tag>/                     # e.g. "bank_revisit"
              videos/                  # rendered eval videos
              config.yaml              # snapshot of the eval config

Design notes
------------
- **Config snapshot** (``snapshot_config``): each run writes the exact config it
  ran with next to its outputs, answering "what config produced this result?".
- **Co-located logs**: training logs live inside the run dir, not a global pile.
- **INDEX ledger** (``append_index``): a per-version, per-(train|eval) running
  markdown table giving a one-line verdict for every eval, for quick scanning.
- **OUTPUT_ROOT override**: point the whole tree elsewhere via the env var (CI,
  scratch disk, a teammate's machine) without touching code.
- **No conflict with legacy flat artifacts**: this layout is namespaced under
  ``<version>/`` and is opt-in; existing v2/v3/v4 outputs (their own OUTPUT_DIR)
  are untouched. New v5 scripts use this; old scripts may adopt it later.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

# Optional: prefer YAML for config snapshots, but never hard-depend on PyYAML.
try:  # pragma: no cover - exercised only when PyYAML is installed
    import yaml  # type: ignore

    _HAVE_YAML = True
except Exception:  # ImportError or any partial-install breakage -> fall back to JSON
    _HAVE_YAML = False


#: Root under which every artifact tree is created. Overridable via ``OUTPUT_ROOT``.
OUTPUT_ROOT: Path = Path(
    os.environ.get("OUTPUT_ROOT", "/home/nvme02/wlx/Memory/outputs")
)


def train_run_dir(version: str, run_name: str) -> Path:
    """Return (creating if needed) the directory for one training run.

    Layout: ``OUTPUT_ROOT/<version>/train/<run_name>/`` with the standard
    subdirectories ``checkpoints/``, ``logs/`` and ``samples/`` pre-created.

    Args:
        version: Pipeline version namespace, e.g. ``"v5"``.
        run_name: Unique run identifier, e.g. from :func:`default_run_name`.

    Returns:
        Path to the run directory (already exists on return).
    """
    run_dir = OUTPUT_ROOT / version / "train" / run_name
    for sub in ("checkpoints", "logs", "samples"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def eval_run_dir(version: str, run_name: str, tag: str) -> Path:
    """Return (creating if needed) the directory for one evaluation.

    Layout: ``OUTPUT_ROOT/<version>/eval/<run_name>/<tag>/`` with a ``videos/``
    subdirectory pre-created.

    Args:
        version: Pipeline version namespace, e.g. ``"v5"``.
        run_name: The run being evaluated (often a training run name).
        tag: Eval scenario / probe name, e.g. ``"bank_revisit"``.

    Returns:
        Path to the eval directory (already exists on return).
    """
    run_dir = OUTPUT_ROOT / version / "eval" / run_name / tag
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(run_dir: Path, config: dict) -> Path:
    """Dump ``config`` next to a run's outputs as a config snapshot.

    Writes ``run_dir/config.yaml`` when PyYAML is available, otherwise
    ``run_dir/config.json``. Records exactly which configuration produced the
    artifacts in ``run_dir``.

    Args:
        run_dir: A directory returned by :func:`train_run_dir` /
            :func:`eval_run_dir` (created if it does not yet exist).
        config: JSON/YAML-serializable mapping of the run's configuration.

    Returns:
        Path to the written snapshot file.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if _HAVE_YAML:
        out_path = run_dir / "config.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        out_path = run_dir / "config.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    return out_path


def append_index(version: str, run_name: str, tag: str, verdict: str) -> None:
    """Append a one-line verdict to the per-version eval INDEX ledger.

    Adds a markdown table row to ``OUTPUT_ROOT/<version>/eval/INDEX.md`` with a
    timestamp, the ``run_name`` / ``tag`` and a one-sentence ``verdict``. The
    file (and a table header) is created on first use.

    Args:
        version: Pipeline version namespace, e.g. ``"v5"``.
        run_name: The run that was evaluated.
        tag: Eval scenario / probe name.
        verdict: One-line human summary of the result.
    """
    index_path = OUTPUT_ROOT / version / "eval" / "INDEX.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Keep the verdict on a single markdown table row.
    safe_verdict = " ".join(str(verdict).splitlines()).replace("|", "\\|")
    row = f"| {ts} | {run_name} | {tag} | {safe_verdict} |\n"

    if not index_path.exists():
        header = (
            f"# Eval INDEX — {version}\n\n"
            "| timestamp | run_name | tag | verdict |\n"
            "| --- | --- | --- | --- |\n"
        )
        with index_path.open("w", encoding="utf-8") as f:
            f.write(header)

    with index_path.open("a", encoding="utf-8") as f:
        f.write(row)


def default_run_name(prefix: str) -> str:
    """Build a timestamped run name, e.g. ``"<prefix>_20260621_101500"``.

    Uses the current local time (real server time when run on the server).

    Args:
        prefix: Short human-readable label for the run.

    Returns:
        ``f"{prefix}_{YYYYMMDD_HHMMSS}"``.
    """
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
