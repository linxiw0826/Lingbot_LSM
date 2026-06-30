"""
merge_diag_shards.py — 合并多卡分片 ideal_inject_diag 的 per_window.csv → 单份 + GO/NO-GO
==========================================================================================

ideal_inject_diag.py 加了 `--shard_index / --shard_count`（按 **revisit case** 全局序号取模
分片）后，6 卡并行诊断会各自写到独立 run_dir（脚本传不同 --tag，如 tierA_ideal_s0..s5），
每个 run_dir 下有自己的 per_window.csv。本脚本把各 shard 的 per_window.csv 合并成一份
（只动 CSV，不搬视频），再重算三臂均值 + GO/NO-GO 判决 → summary.md。

== 与 merge_eval_shards.py 的关系 ==
  - 镜像 merge_eval_shards.py 的 CSV 合并 / merged_note 结构与「纯 stdlib、不依赖 torch」约束。
  - 区别：diag 的 per_window.csv 用 `arm` 列（off / ideal_A / random_A [/ ideal_B]）而非
    eval_v5 的 `memory_mode`，判据是 ideal_A 同时 > off + margin 且 > random_A + margin
    （对齐 ideal_inject_diag._verdict）。故不能复用 summarize_eval.summarize_run，本脚本
    内置一份**与 ideal_inject_diag._verdict 等价**的 torch-free 判决（保持 summary.md 同格式）。

== 数据流（跨模块契约）==
  各 shard_dir/per_window.csv  ──(合并所有数据行)──▶  out_dir/per_window.csv
  out_dir/per_window.csv        ──(本脚本三臂判决)──▶  out_dir/summary.md
  各 shard_dir/videos/          ──(不搬，原位保留)──▶  merged_note.md 记录路径

本地无服务器产出也能跑（手工造几个 per_window.csv 即可验证合并 + 判决逻辑）。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

PER_WINDOW_CSV = "per_window.csv"

# 臂顺序（与 ideal_inject_diag.ALL_ARMS 对齐；off 在前作 baseline）。
ALL_ARMS = ("off", "ideal_A", "random_A", "ideal_B")
# GO 判据 margin（与 ideal_inject_diag --go_margin 默认一致）。
DEFAULT_MARGIN = 0.01
# diag per_window.csv 列（与 ideal_inject_diag._CSV_FIELDS 对齐；全空合并时的兜底 header）。
DIAG_CSV_FIELDS = [
    "episode_id", "query_frame", "first_visit_frame", "arm",
    "weaken_first_frame", "video_path", "gt_first_visit_png",
    "dino_max", "dino_mean", "dino_last",
    "ssim_max", "ssim_mean", "ssim_last",
]


# ---------------------------------------------------------------------------
# CSV 合并（镜像 merge_eval_shards.py）
# ---------------------------------------------------------------------------

def _read_csv_rows(csv_path: str) -> Tuple[Optional[List[str]], List[dict]]:
    """读一个 per_window.csv，返回 (fieldnames, data_rows)。

    文件不存在 / 空（只有 header 或完全空）→ 返回 (None, [])，调用方据此 WARN 跳过。
    优先用 pandas 读，pandas 不可用时退回 csv 模块。
    """
    if not os.path.exists(csv_path):
        return None, []

    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path)
        if df.empty:
            return None, []
        fieldnames = list(df.columns)
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
            print(f"[WARN] shard_dir 的 per_window.csv header 与首个不一致：{sd}",
                  f"\n        首: {fieldnames}\n        此: {fn}", file=sys.stderr)
        all_rows.extend(rows)
        merged_from.append(sd)

    if fieldnames is None:
        print("[ERROR] 所有 shard 均无可用 per_window.csv，合并为空。", file=sys.stderr)
        fieldnames = list(DIAG_CSV_FIELDS)

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    return len(all_rows), merged_from, skipped


# ---------------------------------------------------------------------------
# 三臂判决（torch-free；与 ideal_inject_diag._verdict 等价）
# ---------------------------------------------------------------------------

def _to_float(v) -> Optional[float]:
    """CSV cell → float；空 / 'None' / 不可解析 → None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip()
    if s == "" or s.lower() in ("none", "nan", "null"):
        return None
    try:
        f = float(s)
        return None if math.isnan(f) else f
    except ValueError:
        return None


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def write_verdict(rows: List[dict], out_dir: str, margin: float) -> str:
    """读合并 rows → 逐 case 三臂表 + 均值 + GO/NO-GO，写 out_dir/summary.md，返回 verdict。

    与 ideal_inject_diag._verdict 逻辑一致：
      GO ⇔ ideal_A 的 DINO 均值 > off + margin **且** > random_A + margin。
    """
    # 出现过的臂（按 ALL_ARMS 顺序，off 在前）
    present = {str(r.get("arm", "")).strip() for r in rows}
    arms = [a for a in ALL_ARMS if a in present]

    # 按 case (ep, q) 聚合每臂 dino_mean
    cases: Dict[tuple, Dict[str, float]] = {}
    for r in rows:
        arm = str(r.get("arm", "")).strip()
        dm = _to_float(r.get("dino_mean"))
        if not arm or dm is None:
            continue
        key = (str(r.get("episode_id", "")), str(r.get("query_frame", "")))
        cases.setdefault(key, {})[arm] = dm

    lines: List[str] = []
    lines.append("# v5-KV 理想注入诊断（S-V4 / Step 43）—— GO/NO-GO（合并多分片）\n")
    lines.append(f"判据: GO ⇔ ideal_A 的 DINO 均值 > off + {margin} **且** > random_A + {margin}\n")
    lines.append("（GO = 情况乙 通道+骨干能用 → 修 encoder，不上 LoRA-q；"
                 "NO-GO = 情况甲 通道用不了 → 阶梯 ② LoRA-q / pivot latent-concat）\n")

    header = "| episode | query | " + " | ".join(arms) + " | ideal_A−off | ideal_A−random_A |"
    sep = "|" + "---|" * (len(arms) + 4)
    lines.append("\n## 逐 case（frame-aligned DINO mean）\n")
    lines.append(header)
    lines.append(sep)

    means: Dict[str, List[float]] = {a: [] for a in arms}
    for key in sorted(cases.keys()):
        row = cases[key]
        cells = []
        for a in arms:
            v = row.get(a)
            cells.append("nan" if v is None else f"{v:.4f}")
            if v is not None:
                means[a].append(v)
        d_off = (row.get("ideal_A", float("nan")) - row.get("off", float("nan")))
        d_rnd = (row.get("ideal_A", float("nan")) - row.get("random_A", float("nan")))
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) +
                     f" | {d_off:+.4f} | {d_rnd:+.4f} |")

    mean_off = _mean(means.get("off", []))
    mean_ideal = _mean(means.get("ideal_A", []))
    mean_rnd = _mean(means.get("random_A", []))

    lines.append("\n## 均值\n")
    for a in arms:
        mv = _mean(means.get(a, []))
        lines.append(f"- {a}: {mv:.4f}  (n={len(means.get(a, []))})")

    go = (
        ("ideal_A" in arms and "off" in arms and "random_A" in arms)
        and (mean_ideal == mean_ideal)  # not nan
        and (mean_off == mean_off) and (mean_rnd == mean_rnd)
        and (mean_ideal > mean_off + margin)
        and (mean_ideal > mean_rnd + margin)
    )
    verdict = "GO" if go else "NO-GO"
    route = ("通道+骨干能用（情况乙）→ 修 encoder（理想 KV 当蒸馏目标 / 改训练目标），不上 LoRA-q"
             if go else
             "通道用不了（情况甲）→ 进 OP-5 阶梯 ② backbone q(/k) LoRA / pivot latent-concat")
    lines.append("\n## 判决\n")
    lines.append(f"**{verdict}** — ideal_A={mean_ideal:.4f} vs off={mean_off:.4f} "
                 f"(Δ={mean_ideal - mean_off:+.4f}) vs random_A={mean_rnd:.4f} "
                 f"(Δ={mean_ideal - mean_rnd:+.4f})，margin={margin}")
    lines.append(f"\n**路由**：{route}")

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print("\n" + summary)
    return verdict


# ---------------------------------------------------------------------------
# merged_note.md（镜像 merge_eval_shards.py）
# ---------------------------------------------------------------------------

def write_merged_note(out_dir: str, shard_dirs: List[str], merged_from: List[str],
                      skipped: List[str], n_rows: int) -> str:
    """写 out_dir/merged_note.md，记录合并来源 + 视频在各 shard_dir/videos/ 下。"""
    import datetime

    note_path = os.path.join(out_dir, "merged_note.md")
    lines: List[str] = []
    lines.append("# Merged Diag Shards Note\n\n")
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
    lines.append("<shard_dir>/videos/<episode>/q<query_frame>/{off,ideal_A,random_A}.mp4\n")
    lines.append("                          + gt_first_visit.png\n")
    lines.append("```\n\n")
    for sd in merged_from:
        lines.append(f"- `{sd}/videos/`\n")
    if skipped:
        lines.append("\n## 跳过的 shard\n\n")
        for sd in skipped:
            lines.append(f"- `{sd}`\n")
    lines.append("\n## 合并产物\n\n")
    lines.append("- `per_window.csv`（本目录，合并自上述 shard）\n")
    lines.append("- `summary.md`（三臂均值 + GO/NO-GO 判决）\n")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return note_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "合并多卡分片 ideal_inject_diag 的 per_window.csv → 单份，并重算三臂 GO/NO-GO 判决。"
            "各 shard 的 --tag 应为 <base_tag>_s<index>，本脚本不强制命名，只看 shard_dirs。"
        )
    )
    p.add_argument("--shard_dirs", nargs="+", required=True,
                   help="各分片的诊断 run_dir（含 per_window.csv），空格分隔")
    p.add_argument("--out_dir", required=True,
                   help="合并目标 dir（写 per_window.csv + summary.md + merged_note.md）")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                   help=f"GO 判据 margin（默认 {DEFAULT_MARGIN}，对齐 ideal_inject_diag --go_margin）")
    return p.parse_args()


def main():
    args = _parse_args()

    shard_dirs = [os.path.abspath(sd) for sd in args.shard_dirs]
    out_dir = os.path.abspath(args.out_dir)

    for sd in shard_dirs:
        if not os.path.isdir(sd):
            print(f"[WARN] shard_dir 不存在（将跳过）：{sd}", file=sys.stderr)

    print(f"[merge] 合并 {len(shard_dirs)} 个 shard → {out_dir}")
    n_rows, merged_from, skipped = merge_per_window_csvs(shard_dirs, out_dir)
    print(f"[merge] 写入 {n_rows} 行 per_window.csv；"
          f"有效 shard {len(merged_from)}，跳过 {len(skipped)}")

    note_path = write_merged_note(out_dir, shard_dirs, merged_from, skipped, n_rows)
    print(f"[merge] merged_note.md → {note_path}")

    # 重算三臂判决（读合并后的 per_window.csv）
    _fn, rows = _read_csv_rows(os.path.join(out_dir, PER_WINDOW_CSV))
    print(f"[merge] 重算三臂 GO/NO-GO（margin={args.margin}）...")
    verdict = write_verdict(rows, out_dir, margin=args.margin)
    print("=====================================================")
    print("  合并判决（GO/NO-GO）")
    print("=====================================================")
    print(verdict)
    print("")
    print(f"summary.md: {os.path.join(out_dir, 'summary.md')}")
    print(f"per_window.csv: {os.path.join(out_dir, 'per_window.csv')}")


if __name__ == "__main__":
    main()
