"""
summarize_eval.py — 读 v5 eval run 的 per_window.csv → 人能直接看的 GO/NO-GO 判决
==================================================================================

experiment_design Step 41（S-V2 第二块）配套。读一个 eval run 的 `per_window.csv`，
按 memory_mode（off / oracle / random）分组算 `revisit_consistency_dino_mean`（CSV 里
列名 `dino_mean`）在所有 query 上的均值，算 oracle−off / oracle−random，逐 query 列
oracle−off 的正负计数，给出 GO / NO-GO 判决并写 summary.md + 追加 INDEX.md。

判决（decisions.md「讨论 8」判据：oracle 的 DINO 是否 > off）：
    GO    ⇔ oracle 的 DINO 均值 > off + margin **且** > random + margin（注入有效）
    NO-GO ⇔ 否则（触发兜底 LoRA）

依赖：纯 pandas / stdlib（有 pandas 用 pandas，否则退回 csv 模块），**不依赖 torch**。
既可被 eval_v5 import（summarize_run），也可独立 `python summarize_eval.py <eval_run_dir>` 跑。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from os.path import abspath, dirname, join
from typing import Dict, List, Optional, Tuple

# paths.append_index 用于追加 INDEX.md（纯 stdlib，无 torch）
_PIPELINE_DIR = dirname(dirname(abspath(__file__)))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                            # → src/
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from pipeline.common.paths import append_index  # noqa: E402
except Exception:  # pragma: no cover - append_index 不可用时降级为 no-op
    def append_index(version, run_name, tag, verdict):  # type: ignore
        pass

# DINO 主判据列名（eval_v5 per_window.csv 写的列）
DINO_MEAN_COL = "dino_mean"
MODE_COL = "memory_mode"
QUERY_KEYS = ("episode_id", "query_frame")

# GO 判据 margin：oracle 须超过 off / random 至少这么多（DINO mean）
DEFAULT_MARGIN = 0.01


# ---------------------------------------------------------------------------
# CSV 读取（pandas 优先，否则 csv 模块；不依赖 torch）
# ---------------------------------------------------------------------------

def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    """读 per_window.csv 为 List[dict]（值为原始 str，数值解析交给 _to_float）。"""
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path)
        # NaN → None（避免 float('nan') 污染均值）
        return [
            {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
             for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]
    except ImportError:
        import csv
        with open(csv_path, newline="") as fh:
            return list(csv.DictReader(fh))


def _to_float(v) -> Optional[float]:
    """把 CSV cell 解析为 float；空 / 'None' / 不可解析 → None。"""
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


def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# 核心统计
# ---------------------------------------------------------------------------

def _arm_dino_means(rows: List[Dict]) -> Dict[str, Optional[float]]:
    """按 memory_mode 分组算 DINO mean 在所有 query 上的均值（off/oracle/random 各一个）。"""
    by_mode: Dict[str, List[float]] = {}
    for r in rows:
        mode = str(r.get(MODE_COL, "")).strip()
        d = _to_float(r.get(DINO_MEAN_COL))
        if mode and d is not None:
            by_mode.setdefault(mode, []).append(d)
    return {mode: _mean(vals) for mode, vals in by_mode.items()}


def _per_query_oracle_minus_off(rows: List[Dict]) -> Tuple[int, int, int]:
    """逐 query 比较 oracle vs off 的 DINO mean，返回 (n_positive, n_total, n_negative)。

    n_total = 同时有 oracle 和 off 两臂 DINO 的 query 数；
    n_positive = oracle > off 的 query 数。
    """
    # (episode_id, query_frame) → {mode: dino_mean}
    grouped: Dict[Tuple, Dict[str, float]] = {}
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in QUERY_KEYS)
        mode = str(r.get(MODE_COL, "")).strip()
        d = _to_float(r.get(DINO_MEAN_COL))
        if mode and d is not None:
            grouped.setdefault(key, {})[mode] = d

    n_pos = n_neg = n_total = 0
    for _key, mode_vals in grouped.items():
        if "oracle" in mode_vals and "off" in mode_vals:
            n_total += 1
            if mode_vals["oracle"] > mode_vals["off"]:
                n_pos += 1
            else:
                n_neg += 1
    return n_pos, n_total, n_neg


def compute_verdict(
    rows: List[Dict],
    margin: float = DEFAULT_MARGIN,
) -> Dict:
    """从 per_window rows 算出判决所需的全部数字 + GO/NO-GO。

    Returns dict:
        arm_means        : {off/oracle/random: dino_mean 均值 or None}
        oracle_minus_off : float or None
        oracle_minus_random : float or None
        n_query_pos / n_query_total / n_query_neg : 逐 query oracle>off 计数
        go               : bool
        reason           : str（一句话判决说明）
        margin           : float
    """
    arm_means = _arm_dino_means(rows)
    off = arm_means.get("off")
    oracle = arm_means.get("oracle")
    random_ = arm_means.get("random")

    o_minus_off = (oracle - off) if (oracle is not None and off is not None) else None
    o_minus_rnd = (oracle - random_) if (oracle is not None and random_ is not None) else None

    n_pos, n_total, n_neg = _per_query_oracle_minus_off(rows)

    # GO 判据：oracle > off + margin 且 oracle > random + margin
    go = False
    if oracle is None:
        reason = "NO-GO：缺 oracle 臂 DINO 数据（无法判决）。"
    elif off is None:
        reason = "NO-GO：缺 off 臂 DINO 数据（无法判决）。"
    else:
        beat_off = oracle > off + margin
        beat_rnd = (random_ is None) or (oracle > random_ + margin)
        go = beat_off and beat_rnd
        _off_s = f"{off:.4f}"
        _ora_s = f"{oracle:.4f}"
        _rnd_s = "n/a" if random_ is None else f"{random_:.4f}"
        if go:
            reason = (
                f"GO：oracle DINO={_ora_s} > off={_off_s} "
                f"(+{o_minus_off:.4f}) 且 > random={_rnd_s} "
                f"(margin={margin}); 逐 query oracle>off {n_pos}/{n_total}。"
            )
        else:
            _why = []
            if not beat_off:
                _why.append(f"oracle({_ora_s}) 未超 off({_off_s})+{margin}")
            if not beat_rnd:
                _why.append(f"oracle({_ora_s}) 未超 random({_rnd_s})+{margin}")
            reason = (
                "NO-GO：" + "；".join(_why) +
                f"; 逐 query oracle>off {n_pos}/{n_total}（触发兜底 LoRA）。"
            )

    return {
        "arm_means": arm_means,
        "oracle_minus_off": o_minus_off,
        "oracle_minus_random": o_minus_rnd,
        "n_query_pos": n_pos,
        "n_query_total": n_total,
        "n_query_neg": n_neg,
        "go": go,
        "reason": reason,
        "margin": margin,
    }


# ---------------------------------------------------------------------------
# summary.md 渲染
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


def render_summary_md(v: Dict, run_name: str = "", tag: str = "") -> str:
    am = v["arm_means"]
    lines: List[str] = []
    verdict_tag = "✅ GO" if v["go"] else "❌ NO-GO"
    lines.append(f"# v5 Eval Summary — {verdict_tag}\n\n")
    if run_name or tag:
        lines.append(f"- run: `{run_name}` | tag: `{tag}`\n")
    lines.append(f"- **判决：{v['reason']}**\n\n")
    lines.append("> 判据（decisions.md「讨论 8」）：oracle 的 DINO 均值 "
                 f"> off **且** > random（margin={v['margin']}）→ GO（注入有效）；"
                 "否则 NO-GO（触发兜底 LoRA）。DINO 为主判据。\n\n")

    lines.append("## 各臂 DINO mean 均值（所有 query 平均）\n\n")
    lines.append("| arm | dino_mean (avg over queries) |\n")
    lines.append("|---|---|\n")
    for arm in ("off", "oracle", "random"):
        lines.append(f"| {arm} | {_fmt(am.get(arm))} |\n")
    lines.append("\n")

    lines.append("## 差值（DINO mean）\n\n")
    lines.append("| metric | value |\n|---|---|\n")
    lines.append(f"| oracle − off | {_fmt(v['oracle_minus_off'])} |\n")
    lines.append(f"| oracle − random | {_fmt(v['oracle_minus_random'])} |\n")
    lines.append(
        f"| 逐 query oracle>off | {v['n_query_pos']}/{v['n_query_total']}"
        f"（负 {v['n_query_neg']}） |\n\n"
    )

    lines.append("## 人工复核\n\n")
    lines.append("逐 query 三臂视频同夹：`videos/<episode>/<query>/"
                 "{off,oracle,random}.mp4` + `gt_first_visit.png`。\n")
    lines.append("判读：oracle 是否比 off / random 更贴近 GT 首访帧？"
                 "random 是否 ≈ off（confound 排除）？\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def summarize_run(
    eval_run_dir: str,
    run_name: str = "",
    tag: str = "",
    margin: float = DEFAULT_MARGIN,
    update_index: bool = True,
) -> str:
    """读 <eval_run_dir>/per_window.csv → 写 summary.md + 追加 INDEX.md，返回一句话判决。

    Args:
        eval_run_dir: 含 per_window.csv 的 eval run 目录。
        run_name/tag: 用于 INDEX.md 与 summary 标题（缺省从路径推断）。
        margin:       GO 判据 margin（默认 0.01）。
        update_index: 是否追加 INDEX.md（独立跑时可关）。

    Returns:
        一句话判决（GO/NO-GO + 数字）。
    """
    csv_path = os.path.join(eval_run_dir, "per_window.csv")
    if not os.path.exists(csv_path):
        verdict = f"NO-GO：未找到 per_window.csv（{csv_path}）。"
        with open(os.path.join(eval_run_dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(f"# v5 Eval Summary — ❌ NO-GO\n\n- {verdict}\n")
        return verdict

    # 缺省从路径推断 run_name / tag：.../eval/<run_name>/<tag>/
    if not run_name or not tag:
        _tag = os.path.basename(os.path.normpath(eval_run_dir))
        _rn = os.path.basename(os.path.dirname(os.path.normpath(eval_run_dir)))
        run_name = run_name or _rn
        tag = tag or _tag

    rows = _read_rows(csv_path)
    v = compute_verdict(rows, margin=margin)
    md = render_summary_md(v, run_name=run_name, tag=tag)

    with open(os.path.join(eval_run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    if update_index:
        try:
            append_index("v5", run_name, tag, v["reason"])
        except Exception:  # noqa: BLE001
            pass

    return v["reason"]


def _parse_args():
    p = argparse.ArgumentParser(
        description="读 v5 eval run 的 per_window.csv，写 summary.md + INDEX.md（GO/NO-GO 判决）。")
    p.add_argument("eval_run_dir", type=str,
                   help="含 per_window.csv 的 eval run 目录")
    p.add_argument("--run_name", type=str, default="",
                   help="INDEX/标题用 run 名（默认从路径推断）")
    p.add_argument("--tag", type=str, default="",
                   help="INDEX/标题用 tag（默认从路径推断）")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                   help=f"GO 判据 margin（默认 {DEFAULT_MARGIN}）")
    p.add_argument("--no_index", action="store_true",
                   help="不追加 INDEX.md（独立跑用）")
    return p.parse_args()


def main():
    args = _parse_args()
    verdict = summarize_run(
        args.eval_run_dir,
        run_name=args.run_name,
        tag=args.tag,
        margin=args.margin,
        update_index=not args.no_index,
    )
    print(verdict)


if __name__ == "__main__":
    main()
