"""
merge_eval_shards.py — 合并多卡分片 eval 的 per_window.csv → 单份 + GO/NO-GO 判决
===================================================================================

eval_v5.py 加了 `--shard_index / --shard_count` 后，6 卡并行 eval 会各自写到独立
run_dir（不同 --tag，如 bank_revisit_s0..s5），每个 run_dir 下有自己的 per_window.csv。
本脚本把各 shard 的 per_window.csv 合并成一份（只动 CSV，不搬视频），再调
`summarize_eval.summarize_run` 出 summary.md + INDEX.md。

== 数据流（跨模块契约）==
  各 shard_dir/per_window.csv  ──(本脚本合并所有数据行)──▶  out_dir/per_window.csv
  out_dir/per_window.csv        ──(summarize_eval.summarize_run)──▶  out_dir/summary.md
                                                                  + INDEX.md 追加
  各 shard_dir/videos/          ──(不搬，原位保留)──▶  merged_note.md 记录路径

== 设计约束 ==
  - 纯 stdlib（+ 可选 pandas，对齐 summarize_eval 的读取口径），**不依赖 torch**。
  - 空文件 / shard 缺 per_window.csv → WARN 跳过，不崩（部分 shard 失败也能出判决）。
  - 只保留一个 header（取第一个非空 shard 的 header），其余 shard 的数据行直接续上。

本地无服务器产出也能跑（手工造几个 per_window.csv 即可验证合并逻辑）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from os.path import abspath, dirname, join
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# sys.path（与 summarize_eval.py / eval_v5.py 一致，确保能 import pipeline.v5.summarize_eval）
# ---------------------------------------------------------------------------
_PIPELINE_DIR = dirname(dirname(abspath(__file__)))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                            # → src/
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from pipeline.v5.summarize_eval import summarize_run  # noqa: E402


PER_WINDOW_CSV = "per_window.csv"


# ---------------------------------------------------------------------------
# CSV 合并
# ---------------------------------------------------------------------------

def _read_csv_rows(csv_path: str) -> Tuple[Optional[List[str]], List[dict]]:
    """读一个 per_window.csv，返回 (fieldnames, data_rows)。

    文件不存在 / 空（只有 header 或完全空）→ 返回 (None, [])，调用方据此 WARN 跳过。
    优先用 pandas 读（对齐 summarize_eval 口径），pandas 不可用时退回 csv 模块。
    """
    if not os.path.exists(csv_path):
        return None, []

    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path)
        if df.empty:
            return None, []
        fieldnames = list(df.columns)
        # NaN → 空串，避免 float('nan') 写进合并 CSV
        rows = [
            {k: ("" if (isinstance(v, float) and v != v) else v) for k, v in rec.items()}
            for rec in df.to_dict(orient="records")
        ]
        return fieldnames, rows
    except ImportError:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not rows:
            return None, []
        return fieldnames, rows


def merge_per_window_csvs(shard_dirs: List[str], out_dir: str) -> Tuple[int, List[str], List[str]]:
    """合并各 shard_dir/per_window.csv → out_dir/per_window.csv。

    Returns:
        (n_total_rows, merged_from, skipped)
        n_total_rows : 合并写入的数据行总数。
        merged_from  : 实际贡献了数据行的 shard_dir 列表（用于 merged_note.md）。
        skipped      : 被跳过的 shard_dir 列表（缺文件 / 空文件）。
    """
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, PER_WINDOW_CSV)

    fieldnames: Optional[List[str]] = None
    all_rows: List[dict] = []
    merged_from: List[str] = []
    skipped: List[str] = []

    for sd in shard_dirs:
        csv_path = os.path.join(sd, PER_WINDOW_CSV)
        fn, rows = _read_csv_rows(csv_path)
        if fn is None or not rows:
            print(f"[WARN] shard_dir 缺 per_window.csv 或为空，跳过：{sd}", file=sys.stderr)
            skipped.append(sd)
            continue
        if fieldnames is None:
            fieldnames = fn  # 取第一个非空 shard 的 header
        elif fn != fieldnames:
            # header 不一致：以首个为准，多余列丢弃、缺列补空（DictWriter extrasaction='ignore'
            # 会自动处理多余列；缺列写入时该 cell 为空）。打个 WARN 让人知道。
            print(f"[WARN] shard_dir 的 per_window.csv header 与首个不一致：{sd}",
                  f"\n        首: {fieldnames}\n        此: {fn}", file=sys.stderr)
        all_rows.extend(rows)
        merged_from.append(sd)

    if fieldnames is None:
        # 所有 shard 都没数据 → 仍写一个空 CSV（带 header 占位），让 summarize_run 给出
        # 明确的 NO-GO（缺数据）判决而不是 FileNotFoundError。
        print("[ERROR] 所有 shard 均无可用 per_window.csv，合并为空。", file=sys.stderr)
        # 造一个最小 header（对齐 eval_v5._CSV_FIELDS，保证 summarize_eval 能跑）
        fieldnames = [
            "episode_id", "query_frame", "first_visit_frame", "memory_mode",
            "weaken_first_frame", "video_path", "gt_first_visit_png",
            "dino_max", "dino_mean", "dino_last", "max", "mean", "last",
        ]

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    return len(all_rows), merged_from, skipped


# ---------------------------------------------------------------------------
# merged_note.md
# ---------------------------------------------------------------------------

def write_merged_note(out_dir: str, shard_dirs: List[str], merged_from: List[str],
                      skipped: List[str], n_rows: int) -> str:
    """写 out_dir/merged_note.md，记录合并来源 + 视频在各 shard_dir/videos/ 下。"""
    import datetime

    note_path = os.path.join(out_dir, "merged_note.md")
    lines: List[str] = []
    lines.append("# Merged Eval Shards Note\n\n")
    lines.append(f"- 合并时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"- 合并数据行数: {n_rows}\n")
    lines.append(f"- 输入 shard_dirs ({len(shard_dirs)}):\n")
    for sd in shard_dirs:
        mark = "" if sd in merged_from else "  *(skipped: 缺/空 per_window.csv)*"
        lines.append(f"  - `{sd}`{mark}\n")
    lines.append("\n")
    lines.append("## 视频位置（未搬运，原位保留）\n\n")
    lines.append("合并**只动 per_window.csv**，各 shard 的三臂对比视频仍在各自 shard_dir 下：\n\n")
    lines.append("```text\n")
    lines.append("<shard_dir>/videos/<episode>/q<query_frame>/{off,oracle,random}.mp4\n")
    lines.append("                          + gt_first_visit.png\n")
    lines.append("```\n\n")
    for sd in merged_from:
        lines.append(f"- `{sd}/videos/`\n")
    if skipped:
        lines.append("\n## 跳过的 shard\n\n")
        for sd in skipped:
            lines.append(f"- `{sd}`\n")
    lines.append("\n## 合并产物\n\n")
    lines.append(f"- `per_window.csv`（本目录，合并自上述 shard）\n")
    lines.append("- `summary.md`（summarize_eval.summarize_run 生成，GO/NO-GO 判决）\n")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return note_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "合并多卡分片 eval 的 per_window.csv → 单份，并调 summarize_eval 出 GO/NO-GO 判决。"
            "各 shard 的 --tag 应为 <base_tag>_s<index>，本脚本不强制命名，只看 shard_dirs。"
        )
    )
    p.add_argument("--shard_dirs", nargs="+", required=True,
                   help="各分片的 eval run_dir（含 per_window.csv），空格分隔")
    p.add_argument("--out_dir", required=True,
                   help="合并目标 dir（写 per_window.csv + summary.md + merged_note.md）")
    p.add_argument("--run_name", default="",
                   help="传给 summarize_run 的 run_name（INDEX/标题用，缺省从 out_dir 推断）")
    p.add_argument("--tag", default="",
                   help="传给 summarize_run 的 tag（INDEX/标题用，缺省从 out_dir 推断）")
    p.add_argument("--margin", type=float, default=0.01,
                   help="GO 判据 margin（默认 0.01，对齐 summarize_eval.DEFAULT_MARGIN）")
    return p.parse_args()


def main():
    args = _parse_args()

    shard_dirs = [os.path.abspath(sd) for sd in args.shard_dirs]
    out_dir = os.path.abspath(args.out_dir)

    # 基本校验：shard_dirs 至少存在（不存在只 WARN，merge 函数会再 skip）
    for sd in shard_dirs:
        if not os.path.isdir(sd):
            print(f"[WARN] shard_dir 不存在（将跳过）：{sd}", file=sys.stderr)

    print(f"[merge] 合并 {len(shard_dirs)} 个 shard → {out_dir}")
    n_rows, merged_from, skipped = merge_per_window_csvs(shard_dirs, out_dir)
    print(f"[merge] 写入 {n_rows} 行 per_window.csv；"
          f"有效 shard {len(merged_from)}，跳过 {len(skipped)}")

    # 写 merged_note.md
    note_path = write_merged_note(out_dir, shard_dirs, merged_from, skipped, n_rows)
    print(f"[merge] merged_note.md → {note_path}")

    # 调 summarize_run（读合并后的 per_window.csv → summary.md + INDEX.md）
    print(f"[merge] 调 summarize_eval.summarize_run(out_dir={out_dir}, "
          f"run_name={args.run_name!r}, tag={args.tag!r}, margin={args.margin}) ...")
    verdict = summarize_run(out_dir, run_name=args.run_name, tag=args.tag,
                            margin=args.margin)
    print("=====================================================")
    print("  合并判决（GO/NO-GO）")
    print("=====================================================")
    print(verdict)
    print("")
    print(f"summary.md: {os.path.join(out_dir, 'summary.md')}")
    print(f"per_window.csv: {os.path.join(out_dir, 'per_window.csv')}")


if __name__ == "__main__":
    main()
