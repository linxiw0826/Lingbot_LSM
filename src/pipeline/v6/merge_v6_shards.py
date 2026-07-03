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
import csv
import json
import os
import sys
from collections import OrderedDict
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# 修改 B：合并后去重（同一 (ep, q, arm, win_start, revisit_clip_idx) 只留 DINO 非空的一行）
# ---------------------------------------------------------------------------

def _dedup_key(r: dict) -> Tuple[str, str, str, str, str]:
    return (
        str(r.get("episode_id", "")).strip(),
        str(r.get("query_frame", "")).strip(),
        str(r.get("arm", "")).strip(),
        str(r.get("win_start", "")).strip(),
        str(r.get("revisit_clip_idx", "")).strip(),
    )


def _dedup_rows_keep_dino(rows: List[dict]) -> Tuple[List[dict], int]:
    """按 (ep, q, arm, win_start, revisit_clip_idx) 去重，保留 DINO 非空的行（修改 B）。

    ep05_p09/q560 曾同时落进 shard s1(有 DINO)/s2(DINO 空) → per_window 出现两行、一行有值
    一行空。合并侧治标去重：同 key 多行时，优先留 dino_mean 非空的那行，丢弃空行。
    Returns: (deduped_rows, n_dropped)。保持首次出现顺序。
    """
    by_key: "OrderedDict[tuple, dict]" = OrderedDict()
    dropped = 0
    for r in rows:
        key = _dedup_key(r)
        if key not in by_key:
            by_key[key] = r
            continue
        # 同 key 已存在 → 二选一：留 dino_mean 非空的行
        existing = by_key[key]
        cur_has = _to_float(r.get("dino_mean")) is not None
        old_has = _to_float(existing.get("dino_mean")) is not None
        if cur_has and not old_has:
            by_key[key] = r  # 用非空行替换已存的空行
        dropped += 1
    return list(by_key.values()), dropped


def _rewrite_csv(csv_path: str, fieldnames: Optional[List[str]], rows: List[dict]) -> None:
    """把去重后的 rows 写回 per_window.csv（保持原 header；缺 header 时从 rows 推断）。"""
    if not fieldnames:
        seen: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# 修改 C：status 过滤（GO/NO-GO 统计排除 status != ok 的行）
# ---------------------------------------------------------------------------

def _status_ok(r: dict) -> bool:
    """行是否计入 GO/NO-GO：status 缺失（旧 CSV 向后兼容）或 == ok 才算。"""
    s = str(r.get("status", "")).strip().lower()
    return s in ("", "ok")


# ---------------------------------------------------------------------------
# 修改 D：读 manifest 取每个 case 的 d_vr / d_vrand（天花板强度观测）
# ---------------------------------------------------------------------------

def _load_manifest_map(manifest_path: Optional[str]) -> Dict[Tuple[str, str], dict]:
    """读 cases_manifest.json → {(episode_id, query_frame): {d_vr, d_vrand, keep}}。

    manifest 缺失 / 读失败 → 返回空 dict（summary 的天花板节退化为「无 manifest」，不致命）。
    """
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 读 manifest 失败（{exc}）：{manifest_path}", file=sys.stderr)
        return {}
    out: Dict[Tuple[str, str], dict] = {}
    for c in payload.get("cases", []):
        key = (str(c.get("episode_id", "")), str(c.get("query_frame", "")))
        out[key] = {"d_vr": c.get("d_vr"), "d_vrand": c.get("d_vrand"),
                    "keep": c.get("keep")}
    return out


# ---------------------------------------------------------------------------
# 三口径均值 + 单口径 GO/NO-GO（dino_mean 主 + dino_max / dino_last 辅，修改 D）
# ---------------------------------------------------------------------------

def _aggregate_metric(rows: List[dict], arms: List[str], metric_col: str
                      ) -> Dict[Tuple[str, str], Dict[str, float]]:
    """按 case (ep, q) 聚合每臂某口径 DINO（仅计入 status==ok 且该 metric 非空的行）。"""
    cases: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        if not _status_ok(r):
            continue
        arm = str(r.get("arm", "")).strip()
        if arm not in arms:
            continue
        v = _to_float(r.get(metric_col))
        if v is None:
            continue
        key = (str(r.get("episode_id", "")), str(r.get("query_frame", "")))
        cases.setdefault(key, {})[arm] = v
    return cases


def _metric_verdict(cases: Dict[Tuple[str, str], Dict[str, float]],
                    arms: List[str], margin: float) -> Tuple[str, float, float, float, Dict[str, int]]:
    """单口径三臂均值 + GO/NO-GO。Returns (verdict, mean_off, mean_ideal, mean_rnd, n_by_arm)。"""
    means: Dict[str, List[float]] = {a: [] for a in arms}
    for _key, row in cases.items():
        for a in arms:
            if a in row:
                means[a].append(row[a])
    mean_off = _mean(means.get("off", []))
    mean_ideal = _mean(means.get("anchor_ideal", []))
    mean_rnd = _mean(means.get("anchor_random", []))
    go = (
        ("anchor_ideal" in arms and "off" in arms and "anchor_random" in arms)
        and (mean_ideal == mean_ideal)  # not nan
        and (mean_off == mean_off) and (mean_rnd == mean_rnd)
        and (mean_ideal > mean_off + margin)
        and (mean_ideal > mean_rnd + margin)
    )
    n_by_arm = {a: len(means.get(a, [])) for a in arms}
    return ("GO" if go else "NO-GO"), mean_off, mean_ideal, mean_rnd, n_by_arm


def write_verdict_v6(rows: List[dict], out_dir: str, margin: float,
                     manifest_map: Optional[Dict[Tuple[str, str], dict]] = None) -> str:
    """读合并 rows → 逐 case 三臂表 + 均值 + GO/NO-GO，写 out_dir/summary.md，返回主判据 verdict。

    主判据（与 latentconcat_ideal_diag._verdict 一致，dino_mean 口径）：
      GO ⇔ anchor_ideal 的 DINO 均值 > off + margin **且** > anchor_random + margin。
    修改 C：status != ok 的行（dino_empty / error）排除出所有 GO/NO-GO 统计，并在 summary 报告
      排除了几行、哪些 case。
    修改 D：额外给出 dino_max / dino_last 两个辅助口径 GO/NO-GO（三口径并列，主判据仍 dino_mean）；
      并从 manifest 逐 case 列出 d_vr / d_vrand（天花板强度）。
    """
    manifest_map = manifest_map or {}

    # 修改 C：分离 ok 行与被排除行
    ok_rows = [r for r in rows if _status_ok(r)]
    excluded_rows = [r for r in rows if not _status_ok(r)]

    present = {str(r.get("arm", "")).strip() for r in ok_rows}
    arms = [a for a in ALL_ARMS if a in present]

    # 三口径聚合（仅 ok 行）
    cases_mean = _aggregate_metric(ok_rows, arms, "dino_mean")
    cases_max = _aggregate_metric(ok_rows, arms, "dino_max")
    cases_last = _aggregate_metric(ok_rows, arms, "dino_last")

    lines: List[str] = []
    lines.append("# v6 latent-concat 理想注入诊断（S-V5 / Step 44）—— GO/NO-GO（合并多分片）\n")
    lines.append(f"判据（主）: GO ⇔ anchor_ideal 的 DINO 均值 > off + {margin} **且** "
                 f"> anchor_random + {margin}\n")
    lines.append("（GO = latent-concat 通道多 clip 端到端能用记忆 + 内容特异 → 进 StoryMem 训练；"
                 "NO-GO = 升级更重干预 LoRA/解冻）\n")

    # ---- 修改 C：被排除行报告 ----
    lines.append("\n## 被排除行（status != ok，不计入 GO/NO-GO）\n")
    if excluded_rows:
        lines.append(f"共排除 {len(excluded_rows)} 行：\n")
        lines.append("| episode | query | arm | status |")
        lines.append("|---|---|---|---|")
        for r in excluded_rows:
            lines.append(f"| {r.get('episode_id','')} | {r.get('query_frame','')} | "
                         f"{r.get('arm','')} | {str(r.get('status','')).strip() or '?'} |")
    else:
        lines.append("无（所有行 status=ok 或旧 CSV 无 status 列）。")

    # ---- 逐 case（dino_mean 主口径）----
    header = "| episode | query | " + " | ".join(arms) + " | ideal−off | ideal−random |"
    sep = "|" + "---|" * (len(arms) + 4)
    lines.append("\n## 逐 case（frame-aligned DINO mean，重访 clip vs GT 首访帧；主口径）\n")
    lines.append(header)
    lines.append(sep)
    for key in sorted(cases_mean.keys()):
        row = cases_mean[key]
        cells = ["nan" if row.get(a) is None else f"{row[a]:.4f}" for a in arms]
        d_off = (row.get("anchor_ideal", float("nan")) - row.get("off", float("nan")))
        d_rnd = (row.get("anchor_ideal", float("nan")) - row.get("anchor_random", float("nan")))
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) +
                     f" | {d_off:+.4f} | {d_rnd:+.4f} |")

    # ---- 修改 D：逐 case 天花板强度（d_vr / d_vrand，来自 manifest）----
    lines.append("\n## 逐 case 天花板强度（d_vr / d_vrand，来自 cases_manifest）\n")
    if manifest_map:
        lines.append("d_vr = DINO(GT首访, GT重访)（同地点真值一致性天花板）；"
                     "d_vrand = DINO(GT首访, 随机帧）。d_vr 越高该重访点 GT 层面越强。\n")
        lines.append("| episode | query | d_vr | d_vrand | d_vr−d_vrand |")
        lines.append("|---|---|---|---|---|")
        for key in sorted(cases_mean.keys()):
            m = manifest_map.get(key, {})
            dvr = m.get("d_vr")
            dvrand = m.get("d_vrand")
            s_vr = "nan" if dvr is None else f"{float(dvr):.4f}"
            s_vrand = "nan" if dvrand is None else f"{float(dvrand):.4f}"
            s_delta = ("nan" if (dvr is None or dvrand is None)
                       else f"{float(dvr) - float(dvrand):+.4f}")
            lines.append(f"| {key[0]} | {key[1]} | {s_vr} | {s_vrand} | {s_delta} |")
    else:
        lines.append("（无 manifest 或 manifest 未提供天花板量 → 跳过；传 --manifest 可启用）")

    # ---- 三口径均值 + GO/NO-GO ----
    v_mean, mo_mean, mi_mean, mr_mean, n_mean = _metric_verdict(cases_mean, arms, margin)
    v_max, mo_max, mi_max, mr_max, _n_max = _metric_verdict(cases_max, arms, margin)
    v_last, mo_last, mi_last, mr_last, _n_last = _metric_verdict(cases_last, arms, margin)

    lines.append("\n## 均值（dino_mean 主口径）\n")
    for a in arms:
        lines.append(f"- {a}: {mi_mean if a=='anchor_ideal' else (mo_mean if a=='off' else mr_mean):.4f}"
                     f"  (n={n_mean.get(a, 0)})")

    lines.append("\n## 判决（三口径并列；主判据 = dino_mean）\n")
    lines.append(f"- **dino_mean（主）: {v_mean}** — anchor_ideal={mi_mean:.4f} vs off={mo_mean:.4f} "
                 f"(Δ={mi_mean - mo_mean:+.4f}) vs anchor_random={mr_mean:.4f} "
                 f"(Δ={mi_mean - mr_mean:+.4f})")
    lines.append(f"- dino_max（辅）: {v_max} — anchor_ideal={mi_max:.4f} vs off={mo_max:.4f} "
                 f"(Δ={mi_max - mo_max:+.4f}) vs anchor_random={mr_max:.4f} "
                 f"(Δ={mi_max - mr_max:+.4f})")
    lines.append(f"- dino_last（辅）: {v_last} — anchor_ideal={mi_last:.4f} vs off={mo_last:.4f} "
                 f"(Δ={mi_last - mo_last:+.4f}) vs anchor_random={mr_last:.4f} "
                 f"(Δ={mi_last - mr_last:+.4f})")
    lines.append(f"\n（用户观察：末帧上 ideal 常明显更好但均值被稀释 → dino_last 辅助口径供参考；"
                 f"margin={margin}）")

    verdict = v_mean  # 主判据仍以 dino_mean 为准
    route = ("latent-concat 通道多 clip 端到端能用记忆 + 内容特异 → 进 StoryMem 配方训练"
             "（通道维 concat + negative-RoPE + r128 LoRA on 全 DiT 线性层 + 位姿）"
             if verdict == "GO" else
             "latent-concat 多 clip 端到端不成立 → 升级更重干预（LoRA / 解冻）")
    lines.append("\n## 路由（按主判据 dino_mean）\n")
    lines.append(f"**{verdict}** → {route}")

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
    p.add_argument("--manifest", type=str, default=None,
                   help="（可选，修改 D）cases_manifest.json 路径；提供后 summary 逐 case 列出 "
                        "d_vr/d_vrand 天花板强度。缺省则该节退化为「无 manifest」。")
    return p.parse_args()


def _append_note_section(note_path: str, n_dedup_dropped: int,
                         excluded_rows: List[dict]) -> None:
    """在 merged_note.md 追加去重 + status 排除报告（修改 B/C）。"""
    lines: List[str] = ["\n## 去重 + status 排除（修改 B/C）\n\n"]
    lines.append(f"- 合并后按 (episode_id, query_frame, arm, win_start, revisit_clip_idx) 去重，"
                 f"丢弃 {n_dedup_dropped} 行重复（保留 DINO 非空行）。\n")
    if excluded_rows:
        lines.append(f"- status != ok 排除 {len(excluded_rows)} 行（不计入 GO/NO-GO）：\n")
        for r in excluded_rows:
            lines.append(f"  - ep={r.get('episode_id','')} q={r.get('query_frame','')} "
                         f"arm={r.get('arm','')} status={str(r.get('status','')).strip() or '?'}\n")
    else:
        lines.append("- status != ok 排除 0 行。\n")
    try:
        with open(note_path, "a", encoding="utf-8") as f:
            f.write("".join(lines))
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 追加 merged_note.md 失败: {exc}", file=sys.stderr)


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

    # ---- 修改 B：读回合并 CSV → 去重（同 key 留 DINO 非空行）→ 重写 ----
    merged_csv = os.path.join(out_dir, PER_WINDOW_CSV)
    fieldnames, rows = _read_csv_rows(merged_csv)
    rows, n_dedup_dropped = _dedup_rows_keep_dino(rows)
    if n_dedup_dropped:
        print(f"[merge] 去重丢弃 {n_dedup_dropped} 行重复 (ep,q,arm,win_start,revisit_clip_idx)"
              f"（保留 DINO 非空行）")
    _rewrite_csv(merged_csv, fieldnames, rows)

    # ---- 修改 C：分离 status != ok 行（供 note 报告）----
    excluded_rows = [r for r in rows if not _status_ok(r)]

    note_path = write_merged_note(out_dir, shard_dirs, merged_from, skipped, len(rows))
    _append_note_section(note_path, n_dedup_dropped, excluded_rows)
    print(f"[merge] merged_note.md → {note_path}")

    # ---- 修改 D：读 manifest 取天花板强度 d_vr/d_vrand ----
    manifest_map = _load_manifest_map(args.manifest)

    print(f"[merge] 重算三臂 GO/NO-GO（margin={args.margin}；排除 {len(excluded_rows)} 行 "
          f"status!=ok）...")
    verdict = write_verdict_v6(rows, out_dir, margin=args.margin, manifest_map=manifest_map)
    print("=====================================================")
    print("  v6 合并判决（GO/NO-GO，主判据 dino_mean）")
    print("=====================================================")
    print(verdict)
    print(f"summary.md: {os.path.join(out_dir, 'summary.md')}")
    print(f"per_window.csv: {os.path.join(out_dir, 'per_window.csv')}")


if __name__ == "__main__":
    main()
