#!/usr/bin/env python3
"""compare_3arm.py — 读三臂 eval CSV，打印 trained / trained_none / base 对照 + 增益拆解。

用法：
    python src/scripts/v6/compare_3arm.py \
        /home/nvme02/wlx/Memory/outputs/v6/infer/v6_eval_ep027/action_eval_scores_3arm.csv

拆解（核心）：
    memory  = trained − trained_none   （纯记忆注入增益：权重相同，只差拼不拼 anchor）
    capacity= trained_none − base       （纯 LoRA 容量增益：都不拼 anchor，只差权重）
理想信号：memory > 0 且随 clip 单调放大（记忆随 bank 累积生效）。
"""
import csv
import math
import sys
from collections import defaultdict

ARMS = ["trained", "trained_none", "base"]


def _f(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def _mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")


def main():
    if len(sys.argv) < 2:
        print("用法: python compare_3arm.py <action_eval_scores_3arm.csv>")
        sys.exit(1)
    rows = list(csv.DictReader(open(sys.argv[1])))
    if not rows:
        print("CSV 为空。")
        sys.exit(1)

    dino = defaultdict(list)          # arm -> [dino]
    ssim = defaultdict(list)          # arm -> [ssim]
    dino_clip = defaultdict(lambda: defaultdict(list))   # arm -> clip -> [dino]
    dino_case = defaultdict(lambda: defaultdict(list))   # case -> arm -> [dino]
    for r in rows:
        arm = r["arm"]
        ci = int(r["clip_idx"])
        d = _f(r["dino_mean"])
        s = _f(r["ssim_mean"])
        if d is not None:
            dino[arm].append(d)
            dino_clip[arm][ci].append(d)
            dino_case[r["case"]][arm].append(d)
        if s is not None:
            ssim[arm].append(s)

    clips = sorted({int(r["clip_idx"]) for r in rows})
    present = [a for a in ARMS if a in dino]

    # ---- 各臂：overall DINO / SSIM + 每 clip DINO ----
    print("=" * 78)
    print("各臂 frame-aligned DINO（跨 case 平均）+ SSIM")
    print("=" * 78)
    hdr = f"{'arm':<14}{'DINO':>8}{'SSIM':>8}   " + "".join(f"c{c:<6}" for c in clips)
    print(hdr)
    for arm in present:
        line = f"{arm:<14}{_mean(dino[arm]):>8.4f}{_mean(ssim[arm]):>8.4f}   "
        line += "".join(f"{_mean(dino_clip[arm][c]):<7.3f}" for c in clips)
        print(line)

    # ---- 增益拆解 ----
    def delta(a, b, c):
        return _mean(dino[a][:]) - _mean(dino[b][:]) if a in dino and b in dino else float("nan")

    def delta_clip(a, b, ci):
        if a in dino_clip and b in dino_clip:
            return _mean(dino_clip[a][ci]) - _mean(dino_clip[b][ci])
        return float("nan")

    print("\n" + "=" * 78)
    print("增益拆解（DINO）")
    print("=" * 78)
    if "trained" in dino and "trained_none" in dino:
        line = f"{'memory (t − t_none)':<20}{delta('trained','trained_none',0):>8.4f}   "
        line += "".join(f"{delta_clip('trained','trained_none',c):<7.3f}" for c in clips)
        print(f"{'':<20}{'overall':>8}   " + "".join(f"c{c:<6}" for c in clips))
        print(line)
    if "trained_none" in dino and "base" in dino:
        line = f"{'capacity (t_none − b)':<20}{delta('trained_none','base',0):>8.4f}   "
        line += "".join(f"{delta_clip('trained_none','base',c):<7.3f}" for c in clips)
        print(line)
    if "trained" in dino and "base" in dino:
        line = f"{'total (t − b)':<20}{delta('trained','base',0):>8.4f}   "
        line += "".join(f"{delta_clip('trained','base',c):<7.3f}" for c in clips)
        print(line)

    # ---- 逐 case：memory 增益 ----
    if "trained" in dino and "trained_none" in dino:
        print("\n" + "=" * 78)
        print("逐 case memory 增益（trained − trained_none，overall DINO）")
        print("=" * 78)
        print(f"{'case':<46}{'trained':>9}{'t_none':>9}{'base':>9}{'mem Δ':>9}")
        for case in sorted(dino_case):
            t = _mean(dino_case[case].get("trained", []))
            tn = _mean(dino_case[case].get("trained_none", []))
            b = _mean(dino_case[case].get("base", []))
            print(f"{case:<46}{t:>9.4f}{tn:>9.4f}{b:>9.4f}{t - tn:>9.4f}")

    # ---- 一句话判读 ----
    if "trained" in dino and "trained_none" in dino:
        mem = delta("trained", "trained_none", 0)
        mem_late = _mean([delta_clip("trained", "trained_none", c) for c in clips[len(clips) // 2:]])
        print("\n" + "-" * 78)
        verdict = "记忆有效" if mem > 0 else "记忆无增益/为负"
        trend = "且后段更强（随 clip 放大 ✅）" if mem_late > mem else "但后段未放大（需留意）"
        print(f"判读: memory 增益 overall={mem:+.4f} → {verdict}；后半 clip 均值={mem_late:+.4f} {trend}")


if __name__ == "__main__":
    main()
