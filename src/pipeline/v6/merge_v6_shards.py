"""
merge_v6_shards.py — 合并多卡分片 v6 latentconcat_ideal_diag 的 per_window.csv → 单份 + GO/NO-GO
=================================================================================================

latentconcat_ideal_diag.py 加了 `--shard_index / --shard_count`（按 **revisit case** 全局序号
取模分片）后，多卡并行诊断各自写独立 run_dir（脚本传不同 --tag，如 latentconcat_ideal_s0..s5），
每个 run_dir 下有自己的 per_window.csv。本脚本把各 shard 的 per_window.csv 合并成一份（只动 CSV、
不搬视频），再重算三臂均值 + GO/NO-GO 判决 → summary.md。

**最大化复用**：CSV 合并 / merged_note / 数值解析全部 import merge_diag_shards.py（v5 已有，纯
stdlib、不依赖 torch）。唯一区别 = 臂名（off / anchor_ideal / anchor_random，而非 v5 的
ideal_A / random_A）+ 判据措辞，故只重写一个 **v6 三臂判决**（与
latentconcat_ideal_diag._verdict 等价），其余逻辑直接复用。

判据：GO ⇔ anchor_ideal 的 DINO 均值 > off + margin **且** > anchor_random + margin。
"""

from __future__ import annotations

import argparse
import os
import sys
from os.path import abspath, dirname, join
from typing import Dict, List

# sys.path（让 `pipeline.v5.merge_diag_shards` 可被 import；本脚本可从任意 cwd 跑）
_PIPELINE_DIR = dirname(dirname(abspath(__file__)))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                            # → src/
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# 复用 v5 merge 的 CSV 合并 / note / 数值解析（纯 stdlib，不依赖 torch）
from pipeline.v5.merge_diag_shards import (  # noqa: E402
    PER_WINDOW_CSV,
    _read_csv_rows,
    _to_float,
    _mean,
    merge_per_window_csvs,
    write_merged_note,
)

# v6 臂顺序（off 在前作 baseline）。
ALL_ARMS = ("off", "anchor_ideal", "anchor_random")
DEFAULT_MARGIN = 0.01


def write_verdict_v6(rows: List[dict], out_dir: str, margin: float) -> str:
    """读合并 rows → 逐 case 三臂表 + 均值 + GO/NO-GO，写 out_dir/summary.md，返回 verdict。

    与 latentconcat_ideal_diag._verdict 逻辑一致：
      GO ⇔ anchor_ideal 的 DINO 均值 > off + margin **且** > anchor_random + margin。
    """
    present = {str(r.get("arm", "")).strip() for r in rows}
    arms = [a for a in ALL_ARMS if a in present]

    cases: Dict[tuple, Dict[str, float]] = {}
    for r in rows:
        arm = str(r.get("arm", "")).strip()
        dm = _to_float(r.get("dino_mean"))
        if not arm or dm is None:
            continue
        key = (str(r.get("episode_id", "")), str(r.get("query_frame", "")))
        cases.setdefault(key, {})[arm] = dm

    lines: List[str] = []
    lines.append("# v6 latent-concat 理想注入诊断（S-V5 / Step 44）—— GO/NO-GO（合并多分片）\n")
    lines.append(f"判据: GO ⇔ anchor_ideal 的 DINO 均值 > off + {margin} **且** "
                 f"> anchor_random + {margin}\n")
    lines.append("（GO = latent-concat 通道多 clip 端到端能用记忆 + 内容特异 → 进 StoryMem 训练；"
                 "NO-GO = 升级更重干预 LoRA/解冻）\n")

    header = "| episode | query | " + " | ".join(arms) + " | ideal−off | ideal−random |"
    sep = "|" + "---|" * (len(arms) + 4)
    lines.append("\n## 逐 case（frame-aligned DINO mean，重访 clip vs GT 首访帧）\n")
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
        d_off = (row.get("anchor_ideal", float("nan")) - row.get("off", float("nan")))
        d_rnd = (row.get("anchor_ideal", float("nan")) - row.get("anchor_random", float("nan")))
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) +
                     f" | {d_off:+.4f} | {d_rnd:+.4f} |")

    mean_off = _mean(means.get("off", []))
    mean_ideal = _mean(means.get("anchor_ideal", []))
    mean_rnd = _mean(means.get("anchor_random", []))

    lines.append("\n## 均值\n")
    for a in arms:
        mv = _mean(means.get(a, []))
        lines.append(f"- {a}: {mv:.4f}  (n={len(means.get(a, []))})")

    go = (
        ("anchor_ideal" in arms and "off" in arms and "anchor_random" in arms)
        and (mean_ideal == mean_ideal)  # not nan
        and (mean_off == mean_off) and (mean_rnd == mean_rnd)
        and (mean_ideal > mean_off + margin)
        and (mean_ideal > mean_rnd + margin)
    )
    verdict = "GO" if go else "NO-GO"
    route = ("latent-concat 通道多 clip 端到端能用记忆 + 内容特异 → 进 StoryMem 配方训练"
             "（通道维 concat + negative-RoPE + r128 LoRA on 全 DiT 线性层 + 位姿）"
             if go else
             "latent-concat 多 clip 端到端不成立 → 升级更重干预（LoRA / 解冻）")
    lines.append("\n## 判决\n")
    lines.append(f"**{verdict}** — anchor_ideal={mean_ideal:.4f} vs off={mean_off:.4f} "
                 f"(Δ={mean_ideal - mean_off:+.4f}) vs anchor_random={mean_rnd:.4f} "
                 f"(Δ={mean_ideal - mean_rnd:+.4f})，margin={margin}")
    lines.append(f"\n**路由**：{route}")

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print("\n" + summary)
    return verdict


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "合并多卡分片 v6 latentconcat_ideal_diag 的 per_window.csv → 单份，"
            "并重算三臂 GO/NO-GO 判决（off / anchor_ideal / anchor_random）。"
        )
    )
    p.add_argument("--shard_dirs", nargs="+", required=True,
                   help="各分片的诊断 run_dir（含 per_window.csv），空格分隔")
    p.add_argument("--out_dir", required=True,
                   help="合并目标 dir（写 per_window.csv + summary.md + merged_note.md）")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                   help=f"GO 判据 margin（默认 {DEFAULT_MARGIN}，对齐 --go_margin）")
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

    _fn, rows = _read_csv_rows(os.path.join(out_dir, PER_WINDOW_CSV))
    print(f"[merge] 重算三臂 GO/NO-GO（margin={args.margin}）...")
    verdict = write_verdict_v6(rows, out_dir, margin=args.margin)
    print("=====================================================")
    print("  v6 合并判决（GO/NO-GO）")
    print("=====================================================")
    print(verdict)
    print(f"summary.md: {os.path.join(out_dir, 'summary.md')}")
    print(f"per_window.csv: {os.path.join(out_dir, 'per_window.csv')}")


if __name__ == "__main__":
    main()
