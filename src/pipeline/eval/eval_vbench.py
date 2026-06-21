"""
eval_vbench.py — 多模型 VBench 评测脚本

功能：
  1. 批量推理：对测试集每张图片，调用各模型的推理脚本生成视频
     - 若 YAML 中 video_dir 非空，则跳过推理，直接使用已有视频（demo 模式）
     - 若命令行传入 --video_dir，则直接使用该目录（无 YAML 单次 demo 模式）
  2. VBench 评分：对生成的视频调用 VBench（custom_input 模式）评分
  3. 汇总结果：
     - results_summary.csv（aggregate 分数，每模型一行）
     - results_per_clip.csv（per-clip 分数，每 clip×model 一行）
     - _comparison/all_runs.csv（跨 run 注册表，累积追加）
     - _comparison/comparison_per_clip.csv（跨 run per-clip 对比）
     - _comparison/comparison_aggregate.csv（跨 run aggregate 对比）

模型配置通过 eval_model_configs.yaml 传入（YAML 模式）。
若配置文件不存在，自动生成模板并提示用户填写后重新运行。
无 YAML 单次评测：直接传 --video_dir（demo 模式）或 --infer_script/--ft_model_dir 等参数。

YAML 模板示例（复制到 eval_model_configs.yaml 并填写路径后使用）：
----------------------------------------------------------------------
baseline:
  name: "Baseline"
  video_dir: "outputs/inference/baseline"   # 有值 → demo 模式，直接使用已有视频
  infer_script: ""                           # demo 模式下不需要填
  ckpt_dir: ""

v3_mem:
  name: "v3 + Memory (epoch 5)"
  video_dir: "outputs/inference/v3_stage1_dual_epoch_5_mem"

yume:
  name: "Yume-1.5"
  video_dir: ""         # 空 → full pipeline 模式，走推理生成
  infer_script: "..."
  ckpt_dir: "..."
  launcher: "python"
----------------------------------------------------------------------

用法：
  python eval_vbench.py \\
      --test_images_dir eval_data/images/ \\
      --test_traj_dir   eval_data/trajectories/ \\
      --output_dir      outputs/eval_vbench/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# 日志设置（对齐 infer_v2.py 格式）
# ---------------------------------------------------------------------------

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 项目根目录（用于将相对 video_dir 解析为绝对路径）
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (_SCRIPT_DIR / "../../..").resolve()

# ---------------------------------------------------------------------------
# 默认 VBench 维度（对应论文 Table 2 的 6 个）
# ---------------------------------------------------------------------------

DEFAULT_DIMENSIONS = [
    "imaging_quality",
    "aesthetic_quality",
    "dynamic_degree",
    "motion_smoothness",
    "temporal_flickering",
    "subject_consistency",
]

# ---------------------------------------------------------------------------
# eval_model_configs.yaml 模板
# ---------------------------------------------------------------------------

_YAML_TEMPLATE = """\
# eval_model_configs.yaml — 评测模型配置
# 填写以下字段后运行 eval_vbench.py（或 run_eval.sh）
#
# video_dir 规则：
#   - 非空 → demo 模式，直接使用已有视频，跳过推理
#   - 空或不存在 → full pipeline 模式，走推理生成（需填 infer_script + ckpt_dir）
# 每个 group 独立决定，无需全局 skip_inference。

baseline:
  name: "Baseline"
  video_dir: ""         # 有值则跳过推理，直接使用该目录的已有视频
  infer_script: "src/pipeline/v2/infer_v2.py"
  ckpt_dir: ""          # 基础模型目录（必填，full pipeline 模式）
  lora_path: ""         # LoRA 权重路径（可选，留空则不使用）
  use_memory: false     # 是否启用 Memory Bank
  extra_args: []        # 额外命令行参数列表，示例: ["--sample_steps", "50"]
  launcher: "torchrun"  # 可选 python 或 torchrun（默认 torchrun）
  sp_num_heads: 40   # Wan14B: 40 heads；用于计算最大合法 Ulysses SP GPU 数（heads%k==0）

groupB:
  name: "Yume-1.5"
  video_dir: ""         # 空 → full pipeline 模式
  infer_script: ""      # Yume 推理脚本路径（必填）
  ckpt_dir: ""          # 必填
  extra_args: []
  launcher: "python"    # 或 torchrun

groupC:
  name: "HunyuanVideo-World 1.5"
  video_dir: ""         # 空 → full pipeline 模式
  infer_script: ""      # 必填
  ckpt_dir: ""          # 必填
  extra_args: []
  launcher: "python"    # 或 torchrun
"""

# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="多模型 VBench 批量评测脚本"
    )

    parser.add_argument(
        "--test_images_dir", type=str, default="eval_data/images/",
        help="测试图片目录（.jpg/.png，默认 eval_data/images/）",
    )
    parser.add_argument(
        "--test_traj_dir", type=str, default="eval_data/trajectories/",
        help="相机轨迹目录（与图片同名，后缀不同，默认 eval_data/trajectories/）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/eval_vbench/",
        help="生成视频和评测结果的根目录（默认 outputs/eval_vbench/）",
    )
    parser.add_argument(
        "--model_config", type=str, default="eval_model_configs.yaml",
        help="模型配置 YAML 文件路径（默认 eval_model_configs.yaml）",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=["baseline", "groupB", "groupC"],
        help="要评测的模型 key 列表（默认 baseline groupB groupC）",
    )
    parser.add_argument(
        "--skip_inference", action="store_true", default=False,
        help="跳过所有模型的推理（全局快捷方式），直接对已有视频评分",
    )
    parser.add_argument(
        "--skip_vbench", action="store_true", default=False,
        help="跳过 VBench 评分，只做推理",
    )
    parser.add_argument(
        "--dimensions", type=str, nargs="+", default=DEFAULT_DIMENSIONS,
        help="VBench 评测维度列表（默认覆盖论文 Table 2 的 6 个维度）",
    )
    parser.add_argument(
        "--vbench_mode", type=str, default="custom_input",
        choices=["custom_input"],
        help="VBench 评测模式（目前仅支持 custom_input）",
    )
    parser.add_argument(
        "--frame_num", type=int, default=81,
        help="每个视频生成帧数（默认 81）",
    )
    parser.add_argument(
        "--size", type=str, default="480*832",
        help="分辨率（默认 480*832）",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="First-person view of CS:GO competitive gameplay",
        help="生成 prompt（默认 First-person view of CS:GO competitive gameplay）",
    )

    # ---- 新增：无 YAML 单次评测参数 ----
    parser.add_argument(
        "--video_dir", type=str, default="",
        help="demo 模式：已有视频文件夹绝对路径（非空则跳过推理，直接评分）",
    )
    parser.add_argument(
        "--run_name", type=str, default="",
        help="本次评测的标识名（demo 模式留空则取 basename(video_dir)）",
    )
    parser.add_argument(
        "--infer_script", type=str, default="",
        help="full pipeline 模式：推理脚本路径",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="",
        help="full pipeline 模式：基础模型目录",
    )
    parser.add_argument(
        "--ft_model_dir", type=str, default="",
        help="full pipeline 模式：全参微调/dual-low 模型目录",
    )
    parser.add_argument(
        "--ft_high_model_dir", type=str, default="",
        help="full pipeline 模式：dual-high 模型目录",
    )
    parser.add_argument(
        "--use_memory", action="store_true",
        help="full pipeline 模式：启用 Memory Bank",
    )
    parser.add_argument(
        "--lora_path", type=str, default="",
        help="full pipeline 模式：LoRA 权重路径（可选）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新评测，忽略 all_runs.csv 中的已有记录",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 模型配置加载
# ---------------------------------------------------------------------------

def _load_or_create_model_config(config_path: str) -> dict:
    """加载模型配置 YAML；若不存在则生成模板并退出。"""
    if not os.path.exists(config_path):
        logger.warning(
            f"模型配置文件 '{config_path}' 不存在，正在自动生成模板..."
        )
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(_YAML_TEMPLATE)
        logger.info(
            f"模板已生成：{config_path}\n"
            f"请填写各模型的 ckpt_dir、infer_script 等字段后重新运行。"
        )
        sys.exit(0)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        logger.error(f"配置文件 '{config_path}' 内容为空，请填写后重新运行。")
        sys.exit(1)

    return config


# ---------------------------------------------------------------------------
# 测试集加载
# ---------------------------------------------------------------------------

def _collect_test_images(images_dir: str, traj_dir: str) -> list:
    """收集测试集图片和对应轨迹文件，返回 [(img_path, traj_path), ...] 列表。"""
    images_dir = Path(images_dir)
    traj_dir = Path(traj_dir)

    if not images_dir.exists():
        logger.error(f"测试图片目录不存在：{images_dir}")
        sys.exit(1)

    if not traj_dir.exists():
        logger.warning(f"相机轨迹目录不存在：{traj_dir}，将跳过所有需要轨迹的样本")
        # 不 exit，因为某些模型可能不需要轨迹，这里只是 warning

    image_exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_exts
    ])

    if not image_files:
        logger.error(f"测试图片目录中没有图片：{images_dir}")
        sys.exit(1)

    pairs = []
    for img_path in image_files:
        # 查找同名轨迹文件（后缀不限）
        traj_path = None
        if traj_dir.exists():
            for candidate in traj_dir.iterdir():
                if candidate.stem == img_path.stem:
                    traj_path = candidate
                    break

        if traj_path is None:
            logger.warning(
                f"找不到图片 '{img_path.name}' 对应的轨迹文件，跳过该样本"
            )
            continue

        pairs.append((img_path, traj_path))

    logger.info(f"共找到 {len(pairs)} 对有效测试样本")
    return pairs


# ---------------------------------------------------------------------------
# Clip 名称归一化
# ---------------------------------------------------------------------------

# 时间戳后缀模式：_v<数字>_YYYYMMDD_HHMMSS
_TIMESTAMP_SUFFIX_RE = re.compile(r'_v[0-9]+_\d{8}_\d{6}$')


def _normalize_clip_name(filename: str) -> str:
    """归一化 clip 名称：去掉 .mp4 后缀和时间戳后缀。

    示例：
      "clip001_v3_20260421_153000.mp4" → "clip001"
      "clip001.mp4"                   → "clip001"
      "clip001"                       → "clip001"
      "/path/to/clip001_v3_20260421_153000.mp4" → "clip001"
    """
    name = Path(filename).name   # 取纯文件名，防止完整路径输入
    # 去掉 .mp4 后缀（大小写不敏感）
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    # 去掉时间戳后缀
    name = _TIMESTAMP_SUFFIX_RE.sub("", name)
    return name


def _get_gpu_ids() -> list:
    """从 CUDA_VISIBLE_DEVICES 解析可用 GPU ID 列表。"""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd or cvd in ("NoDevFiles", "-1"):
        return [0]
    try:
        return [int(x.strip()) for x in cvd.split(",") if x.strip()]
    except ValueError:
        return [0]


def _find_max_sp_gpus(n: int, num_heads: int = 40) -> int:
    """找最大合法 GPU 数 k ≤ n，满足 num_heads % k == 0（Ulysses SP 整除约束）。"""
    for k in range(n, 0, -1):
        if num_heads % k == 0:
            return k
    return 1


# ---------------------------------------------------------------------------
# 单模型批量推理
# ---------------------------------------------------------------------------

def _run_inference_for_model(
    model_key: str,
    model_cfg: dict,
    test_pairs: list,
    output_dir: Path,
    args,
    gpu_ids: list = None,
) -> Optional[Path]:
    """对单个模型跑所有测试样本的推理，返回该模型的视频输出目录。

    若 model_cfg['video_dir'] 非空，则为 demo 模式：
      - 解析路径（相对 PROJECT_ROOT），检查存在性
      - 直接返回该路径（不跑推理）
      - 若目录不存在则 logger.error 并返回 None

    否则为 full pipeline 模式，走推理逻辑，生成视频到 output_dir/{model_key}/。
    """
    model_name = model_cfg.get("name", model_key)

    # ---- demo 模式检查 ----
    video_dir_cfg = model_cfg.get("video_dir", "")
    if video_dir_cfg:
        # 解析绝对路径
        vd_path = Path(video_dir_cfg)
        if not vd_path.is_absolute():
            vd_path = PROJECT_ROOT / vd_path
        vd_path = vd_path.resolve()

        if not vd_path.exists():
            logger.error(
                f"[{model_name}] demo 模式：video_dir 指定的目录不存在：{vd_path}"
            )
            return None

        logger.info(
            f"[{model_name}] demo 模式：直接使用已有视频目录 {vd_path}，跳过推理。"
        )
        return vd_path

    # ---- full pipeline 模式 ----
    model_video_dir = output_dir / model_key
    model_video_dir.mkdir(parents=True, exist_ok=True)

    infer_script = model_cfg.get("infer_script", "")
    ckpt_dir = model_cfg.get("ckpt_dir", "")
    lora_path = model_cfg.get("lora_path", "")
    ft_model_dir      = model_cfg.get("ft_model_dir", "")
    ft_high_model_dir = model_cfg.get("ft_high_model_dir", "")
    use_memory = model_cfg.get("use_memory", False)
    extra_args = model_cfg.get("extra_args", [])

    if not infer_script:
        logger.error(
            f"模型 '{model_key}' 的 infer_script 未配置，跳过推理。"
        )
        return model_video_dir

    total = len(test_pairs)
    logger.info(
        f"[{model_name}] 开始推理，共 {total} 张图片，"
        f"输出目录：{model_video_dir}"
    )

    # 解析可用 GPU 列表（full pipeline 模式）
    if gpu_ids is None:
        gpu_ids = _get_gpu_ids()

    for idx, (img_path, traj_path) in enumerate(test_pairs, start=1):
        output_video = model_video_dir / f"{img_path.stem}.mp4"

        # 断点续跑：已存在则跳过
        if output_video.exists():
            logger.info(
                f"[{model_name}] ({idx}/{total}) 已存在，跳过：{output_video.name}"
            )
            continue

        logger.info(
            f"[{model_name}] ({idx}/{total}) 推理：{img_path.name} → {output_video.name}"
        )

        # 拼接推理命令，根据 launcher 决定使用 torchrun 还是 python
        launcher = model_cfg.get("launcher", "torchrun")
        env = os.environ.copy()
        if launcher == "torchrun":
            sp_num_heads = model_cfg.get("sp_num_heads", 40)
            k = _find_max_sp_gpus(len(gpu_ids), sp_num_heads)
            infer_gpu_ids_str = ",".join(str(g) for g in gpu_ids[:k])
            env["CUDA_VISIBLE_DEVICES"] = infer_gpu_ids_str
            cmd = [
                "torchrun", f"--nproc_per_node={k}",
                str(infer_script),
            ]
            logger.info(
                f"[{model_name}] torchrun {k}/{len(gpu_ids)} GPU(s) (SP={sp_num_heads} heads)"
            )
        else:
            infer_gpu_ids_str = str(gpu_ids[0])
            env["CUDA_VISIBLE_DEVICES"] = infer_gpu_ids_str
            cmd = [
                "python",
                str(infer_script),
            ]
        cmd += [
            "--ckpt_dir", str(ckpt_dir),
            "--image", str(img_path),
            "--action_path", str(traj_path),
            "--save_file", str(output_video),
            "--prompt", args.prompt,
            "--frame_num", str(args.frame_num),
            "--size", args.size,
        ]

        # LoRA 权重（组A专有，组B/C通过 extra_args 覆盖）
        if lora_path:
            cmd += ["--lora_path", str(lora_path)]

        # 全参微调 / dual 模型目录（组A专有）
        if ft_model_dir:
            cmd += ["--ft_model_dir", str(ft_model_dir)]
        if ft_high_model_dir:
            cmd += ["--ft_high_model_dir", str(ft_high_model_dir)]

        # Memory Bank（组A专有）
        if use_memory:
            cmd += ["--use_memory"]

        # 额外参数（组B/C用此覆盖命令行格式）
        if extra_args:
            cmd += [str(a) for a in extra_args]

        logger.info(f"执行命令：{' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"[{model_name}] ({idx}/{total}) 推理失败（returncode={e.returncode}），"
                f"跳过：{img_path.name}"
            )
            continue
        except Exception as e:
            logger.warning(
                f"[{model_name}] ({idx}/{total}) 推理异常：{e}，跳过：{img_path.name}"
            )
            continue

    logger.info(f"[{model_name}] 推理完成，视频保存至：{model_video_dir}")
    return model_video_dir


# ---------------------------------------------------------------------------
# VBench 评分
# ---------------------------------------------------------------------------

def _check_vbench_installed():
    """检查 VBench 是否已安装；未安装则打印提示并退出。"""
    try:
        import importlib
        importlib.import_module("vbench")
    except ImportError:
        logger.error(
            "VBench 未安装。请按以下步骤安装：\n"
            "  pip install vbench\n"
            "或参考官方文档：https://github.com/Vchitect/VBench"
        )
        sys.exit(1)


def _run_vbench_single_dim(
    model_key: str,
    model_name: str,
    video_dir: Path,
    dim: str,
    vbench_mode: str,
    output_dir: Path,
    gpu_id: str,
) -> tuple:
    """对单个模型的单个 VBench 维度评分，在指定 GPU 上运行。
    返回 (dim_score, dim_clip_scores)。
    """
    vbench_result_dir = output_dir / "vbench_results" / model_key / dim
    vbench_result_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    cmd = [
        "vbench", "evaluate",
        "--videos_path", str(video_dir),
        "--dimension", dim,
        "--mode", vbench_mode,
        "--output_path", str(vbench_result_dir),
    ]

    logger.info(
        f"[{model_name}] VBench 维度：{dim}（GPU {gpu_id}），命令：{' '.join(cmd)}"
    )

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"[{model_name}] VBench 维度 '{dim}' 失败（returncode={e.returncode}），跳过。"
        )
        return None, {}

    score = _parse_vbench_result(vbench_result_dir, dim)
    clip_scores = _parse_vbench_per_clip(vbench_result_dir, dim)
    logger.info(f"[{model_name}] {dim}: {score}（GPU {gpu_id}）")
    return score, clip_scores


def _run_vbench_for_model(
    model_key: str,
    model_name: str,
    video_dir: Path,
    dimensions: list,
    vbench_mode: str,
    output_dir: Path,
    gpu_id: str = "0",
) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Optional[float]]]]:
    """对单个模型的视频目录跑所有维度的 VBench 评分。

    返回 (aggregate_scores, per_clip_scores)：
      - aggregate_scores: {dimension: score}
      - per_clip_scores:  {dimension: {clip_name: score}}
    """
    scores: Dict[str, Optional[float]] = {}
    per_clip_scores: Dict[str, Dict[str, Optional[float]]] = {}

    for dim in dimensions:
        score, clip_scores = _run_vbench_single_dim(
            model_key, model_name, video_dir, dim, vbench_mode, output_dir, gpu_id
        )
        scores[dim] = score
        per_clip_scores[dim] = clip_scores

    return scores, per_clip_scores


def _parse_vbench_result(result_dir: Path, dimension: str) -> Optional[float]:
    """从 VBench 输出目录中解析指定维度的 aggregate 分数。

    VBench custom_input 模式通常输出 <dimension>_results.json，
    其结构为 {"<dimension>": [[score, ...], total_score], ...}。
    """
    # VBench 不同版本输出文件名可能有差异，尝试多个候选路径
    candidates = [
        result_dir / f"{dimension}_eval_results.json",
        result_dir / f"{dimension}_results.json",
        result_dir / f"results_{dimension}.json",
        result_dir / "results.json",
    ]
    # VBench custom_input 模式实际保存为 results_<timestamp>_eval_results.json
    candidates += sorted(result_dir.glob("results_*_eval_results.json"), reverse=True)

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # custom_input 模式格式：{"video_results": [...], "dimension_score": 0.xxxx}
                if "dimension_score" in data:
                    return float(data["dimension_score"])

                # VBench custom_input 格式：{dimension: [aggregate_score, [per_video_list]]}
                if dimension in data:
                    value = data[dimension]
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, list) and len(value) >= 1:
                        # index 0 = aggregate, index 1 = per-video list
                        if isinstance(value[0], (int, float)):
                            return float(value[0])
                        # 旧版格式：[[per_video_scores], aggregate_score]
                        elif len(value) >= 2 and isinstance(value[1], (int, float)):
                            return float(value[1])

                # 备用：直接取 "score" 键
                if "score" in data:
                    return float(data["score"])

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"解析 VBench 结果文件失败：{candidate}，错误：{e}"
                )
                continue

    logger.warning(
        f"未找到维度 '{dimension}' 的 VBench 结果文件，目录：{result_dir}"
    )
    return None


def _parse_vbench_per_clip(
    result_dir: Path, dimension: str
) -> Dict[str, Optional[float]]:
    """从 VBench 输出 JSON 中解析 per-video 分数。

    支持两种 video_results 格式：
      1. {"video_results": [[filename, score], ...], "dimension_score": 0.xxx}
      2. {"video_results": {"filename.mp4": score, ...}}

    返回 {clip_name: score} 字典（key 为 _normalize_clip_name 后的结果）。
    若 video_results 不存在或解析失败，返回空字典（graceful degrade）。
    """
    candidates = [
        result_dir / f"{dimension}_eval_results.json",
        result_dir / f"{dimension}_results.json",
        result_dir / f"results_{dimension}.json",
        result_dir / "results.json",
    ]
    # VBench custom_input 模式实际保存为 results_<timestamp>_eval_results.json
    candidates += sorted(result_dir.glob("results_*_eval_results.json"), reverse=True)

    for candidate in candidates:
        if not candidate.exists():
            continue

        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)

            clip_scores: Dict[str, Optional[float]] = {}

            # VBench custom_input 格式：{dimension: [aggregate_score, [{video_path, video_results}, ...]]}
            if dimension in data:
                dim_value = data[dimension]
                if isinstance(dim_value, list) and len(dim_value) >= 2 and isinstance(dim_value[1], list):
                    for entry in dim_value[1]:
                        if isinstance(entry, dict):
                            filename = str(entry.get("video_path", ""))
                            try:
                                score = float(entry.get("video_results", None))
                            except (TypeError, ValueError):
                                score = None
                            clip_name = _normalize_clip_name(filename)
                            clip_scores[clip_name] = score
                    if clip_scores:
                        return clip_scores

            # 备用：顶层 video_results 键
            video_results = data.get("video_results")
            if video_results is None:
                continue

            if isinstance(video_results, list):
                # 格式: [[filename, score], ...]
                for entry in video_results:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        filename = str(entry[0])
                        try:
                            score = float(entry[1])
                        except (TypeError, ValueError):
                            score = None
                        clip_name = _normalize_clip_name(filename)
                        clip_scores[clip_name] = score

            elif isinstance(video_results, dict):
                # 格式: {"filename.mp4": score, ...}
                for filename, raw_score in video_results.items():
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        score = None
                    clip_name = _normalize_clip_name(str(filename))
                    clip_scores[clip_name] = score

            return clip_scores

        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            logger.warning(
                f"解析 per-clip 结果失败：{candidate}，错误：{e}"
            )
            continue

    return {}


# ---------------------------------------------------------------------------
# Per-clip 汇总
# ---------------------------------------------------------------------------

def _summarize_per_clip_results(
    all_per_clip: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    model_configs: dict,
    dimensions: list,
    output_dir: Path,
    model_video_dirs: Dict[str, Optional[Path]],
):
    """将 per-clip 评分汇总为 results_per_clip.csv。

    参数：
        all_per_clip:      {model_key: {dim: {clip_name: score}}}
        model_configs:     {model_key: {..., name: ...}}
        dimensions:        维度列表
        output_dir:        输出根目录
        model_video_dirs:  {model_key: video_dir_path}，用于构造 video_path 列

    输出列：clip_name, model_key, model_name, {dim1_score, dim2_score, ...}, video_path

    若 all_per_clip 全为空，不写 CSV，仅打印 info 说明原因。
    """
    # 检查是否有任何 per-clip 数据
    has_any_data = any(
        any(clip_dict for clip_dict in dim_dict.values())
        for dim_dict in all_per_clip.values()
    )
    if not has_any_data:
        logger.info(
            "per-clip 数据全为空（VBench 可能不支持 per-clip 输出或全部解析失败），"
            "跳过 results_per_clip.csv 写入。"
        )
        return

    csv_path = output_dir / "results_per_clip.csv"

    # 收集所有 (model_key, clip_name) 组合
    rows = []
    for model_key, dim_dict in all_per_clip.items():
        model_name = model_configs.get(model_key, {}).get("name", model_key)
        video_dir = model_video_dirs.get(model_key)

        # 收集该模型下所有出现的 clip 名
        all_clip_names: set = set()
        for clip_dict in dim_dict.values():
            all_clip_names.update(clip_dict.keys())

        for clip_name in sorted(all_clip_names):
            # 构造 video_path
            video_path = _resolve_video_path(clip_name, video_dir)

            row = {
                "clip_name": clip_name,
                "model_key": model_key,
                "model_name": model_name,
                "video_path": str(video_path) if video_path else "",
            }

            for dim in dimensions:
                dim_clip_dict = dim_dict.get(dim, {})
                score = dim_clip_dict.get(clip_name)
                row[f"{dim}_score"] = f"{score:.4f}" if score is not None else "N/A"

            rows.append(row)

    if not rows:
        logger.info("per-clip 数据为空，跳过 results_per_clip.csv 写入。")
        return

    # 写 CSV
    fieldnames = (
        ["clip_name", "model_key", "model_name"]
        + [f"{dim}_score" for dim in dimensions]
        + ["video_path"]
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Per-clip CSV 已保存：{csv_path}（共 {len(rows)} 行）")


def _resolve_video_path(
    clip_name: str, video_dir: Optional[Path]
) -> Optional[Path]:
    """尝试定位 clip 对应的视频文件路径。

    策略：
    1. 精确路径：{video_dir}/{clip_name}.mp4 若存在则返回
    2. Glob 匹配：在 video_dir 中找 *{clip_name}*.mp4，取第一个匹配（支持带时间戳文件名）
    3. 均不匹配：返回推断路径 {video_dir}/{clip_name}.mp4（可能不存在）
    """
    if video_dir is None:
        return None

    exact = video_dir / f"{clip_name}.mp4"
    if exact.exists():
        return exact

    # glob 匹配（支持带时间戳后缀的文件名）
    try:
        matches = sorted(video_dir.glob(f"*{clip_name}*.mp4"))
        if matches:
            return matches[0]
    except Exception:
        pass

    # 推断路径
    return exact


# ---------------------------------------------------------------------------
# 结果汇总（aggregate）
# ---------------------------------------------------------------------------

def _summarize_results(
    all_scores: dict,
    model_configs: dict,
    dimensions: list,
    output_dir: Path,
):
    """将所有模型的评分汇总为 CSV 并打印 Markdown 对比表格。

    参数：
        all_scores: {model_key: {dimension: score}}
        model_configs: {model_key: {..., name: ...}}
        dimensions: 维度列表
        output_dir: 输出根目录
    """
    csv_path = output_dir / "results_summary.csv"

    # ---- 写 CSV ----
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Model"] + dimensions
        writer.writerow(header)

        for model_key, dim_scores in all_scores.items():
            model_name = model_configs.get(model_key, {}).get("name", model_key)
            row = [model_name]
            for dim in dimensions:
                score = dim_scores.get(dim)
                row.append(f"{score:.4f}" if score is not None else "N/A")
            writer.writerow(row)

    logger.info(f"CSV 汇总已保存：{csv_path}")

    # ---- 打印 Markdown 表格（对齐论文 Table 2 格式）----
    _print_markdown_table(all_scores, model_configs, dimensions)


def _print_markdown_table(
    all_scores: dict,
    model_configs: dict,
    dimensions: list,
):
    """打印 Markdown 格式对比表格。"""
    # 列宽计算
    model_col_width = max(
        len("Model"),
        max((len(model_configs.get(k, {}).get("name", k)) for k in all_scores), default=5),
    )
    dim_col_widths = {
        dim: max(len(dim), 6) for dim in dimensions
    }

    # 表头
    header_parts = [f"{'Model':<{model_col_width}}"]
    for dim in dimensions:
        header_parts.append(f"{dim:^{dim_col_widths[dim]}}")
    header_line = " | ".join(header_parts)

    separator_parts = ["-" * model_col_width]
    for dim in dimensions:
        separator_parts.append("-" * dim_col_widths[dim])
    separator_line = "-+-".join(separator_parts)

    print("\n" + "=" * len(header_line))
    print("  VBench 评测结果（Table 2 对齐格式）")
    print("=" * len(header_line))
    print("| " + header_line + " |")
    print("|-" + separator_line + "-|")

    for model_key, dim_scores in all_scores.items():
        model_name = model_configs.get(model_key, {}).get("name", model_key)
        row_parts = [f"{model_name:<{model_col_width}}"]
        for dim in dimensions:
            score = dim_scores.get(dim)
            score_str = f"{score:.4f}" if score is not None else "N/A"
            row_parts.append(f"{score_str:^{dim_col_widths[dim]}}")
        print("| " + " | ".join(row_parts) + " |")

    print("=" * len(header_line) + "\n")


# ---------------------------------------------------------------------------
# 跨 run 对比注册表
# ---------------------------------------------------------------------------

def _update_comparison_files(comparison_dir: Path, all_runs_csv: Path) -> None:
    """从 all_runs.csv 重新生成 comparison_per_clip.csv 和 comparison_aggregate.csv。

    纯标准库实现，不依赖 pandas。
    """
    if not all_runs_csv.exists():
        return

    with open(all_runs_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    # 有序去重（保持首次出现顺序）
    def _unique(seq):
        seen: set = set()
        return [x for x in seq if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

    run_names  = _unique(r["run_name"]  for r in rows)
    clip_ids   = _unique(r["clip_id"]   for r in rows)
    dimensions = _unique(r["dimension"] for r in rows)

    # ---- comparison_per_clip.csv ----
    per_clip_data: dict = {}
    for row in rows:
        key = (row["clip_id"], row["dimension"])
        per_clip_data.setdefault(key, {})[row["run_name"]] = row["score"]

    per_clip_path = comparison_dir / "comparison_per_clip.csv"
    with open(per_clip_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "dimension"] + run_names)
        writer.writeheader()
        for clip_id in clip_ids:
            for dim in dimensions:
                key = (clip_id, dim)
                if key not in per_clip_data:
                    continue
                row_out: dict = {"clip_id": clip_id, "dimension": dim}
                for rn in run_names:
                    row_out[rn] = per_clip_data[key].get(rn, "")
                writer.writerow(row_out)
    logger.info(f"已更新 {per_clip_path}")

    # ---- comparison_aggregate.csv ----
    agg: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        s = row["score"]
        if s and s.lower() not in ("", "none", "null"):
            try:
                agg[row["run_name"]][row["dimension"]].append(float(s))
            except ValueError:
                pass

    agg_path = comparison_dir / "comparison_aggregate.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + dimensions + ["overall"])
        writer.writeheader()
        for rn in run_names:
            row_out = {"model": rn}
            dim_avgs = []
            for dim in dimensions:
                vals = agg[rn][dim]
                if vals:
                    avg = sum(vals) / len(vals)
                    row_out[dim] = f"{avg:.4f}"
                    dim_avgs.append(avg)
                else:
                    row_out[dim] = ""
            row_out["overall"] = f"{sum(dim_avgs)/len(dim_avgs):.4f}" if dim_avgs else ""
            writer.writerow(row_out)
    logger.info(f"已更新 {agg_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ---- run_name 推导 ----
    run_name = args.run_name
    if not run_name and args.video_dir:
        run_name = Path(args.video_dir).name
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 构造 synthetic model_configs（优先级：video_dir > infer_script > model_config YAML）----
    if args.video_dir or args.infer_script or args.ft_model_dir:
        # 单模型模式：用 run_name 作为 model_key，不读 YAML
        model_configs = {
            run_name: {
                "name": run_name,
                "video_dir": args.video_dir,
                "infer_script": args.infer_script,
                "ckpt_dir": args.ckpt_dir,
                "ft_model_dir": args.ft_model_dir,
                "ft_high_model_dir": args.ft_high_model_dir,
                "use_memory": args.use_memory,
                "lora_path": args.lora_path,
                "launcher": "torchrun",
                "sp_num_heads": 40,
            }
        }
        model_keys = [run_name]
    else:
        # 原有 YAML 模式（向后兼容）
        model_configs = _load_or_create_model_config(args.model_config)
        model_keys = list(model_configs.keys())
        if not run_name:
            run_name = model_keys[0] if model_keys else datetime.now().strftime("%Y%m%d_%H%M%S")

    # 解析可用 GPU 列表
    gpu_ids = _get_gpu_ids()
    logger.info(f"检测到 GPU: {gpu_ids}（共 {len(gpu_ids)} 张）")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- _comparison 目录 ----
    comparison_dir = Path(args.output_dir).parent / "_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    all_runs_csv = comparison_dir / "all_runs.csv"

    # ---- skip-if-exists ----
    if all_runs_csv.exists():
        with open(all_runs_csv, newline="") as _f:
            _existing_rows = list(csv.DictReader(_f))
        _existing = {row["run_name"] for row in _existing_rows}
        if run_name in _existing:
            if not args.force:
                logger.info(
                    f"[{run_name}] 已在 _comparison/all_runs.csv 中存在，跳过本次评测。"
                    "（传入 --force 可强制重新评测）"
                )
                return
            # --force：先删除旧行，再重新评测
            kept = [r for r in _existing_rows if r["run_name"] != run_name]
            with open(all_runs_csv, "w", newline="") as _f:
                writer = csv.DictWriter(_f, fieldnames=["run_name", "clip_id", "dimension", "score"])
                writer.writeheader()
                writer.writerows(kept)
            logger.info(f"[{run_name}] --force：已从 all_runs.csv 删除旧记录，重新评测。")

    # 过滤请求的模型（无 YAML 单次模式：model_keys 已在上方构造）
    if args.video_dir or args.infer_script or args.ft_model_dir:
        available_models = model_keys
    else:
        requested_models = args.models
        available_models = [k for k in requested_models if k in model_configs]
        missing_models = [k for k in requested_models if k not in model_configs]
        if missing_models:
            logger.warning(
                f"以下模型在配置文件中不存在，已跳过：{missing_models}"
            )
        if not available_models:
            logger.error("没有有效的模型可以评测，请检查 --models 和配置文件。")
            sys.exit(1)

    logger.info(
        f"将评测的模型：{[model_configs[k].get('name', k) for k in available_models]}"
    )

    # ---- Step 1：收集测试集（仅在需要 full pipeline 推理时加载）----
    # 判断是否有任何模型需要 full pipeline 推理
    need_inference = (
        not args.skip_inference
        and any(
            not model_configs[k].get("video_dir", "")
            for k in available_models
        )
    )

    if need_inference:
        test_pairs = _collect_test_images(args.test_images_dir, args.test_traj_dir)
    else:
        test_pairs = []
        logger.info("所有模型均为 demo 模式或已设置 --skip_inference，跳过测试集加载。")

    # ---- Step 2：批量推理（每个模型独立决定 demo / full pipeline）----
    model_video_dirs: Dict[str, Optional[Path]] = {}

    if args.skip_inference:
        logger.info("--skip_inference 已设置，跳过所有模型推理阶段，使用已有视频。")
        for model_key in available_models:
            model_cfg = model_configs[model_key]
            video_dir_cfg = model_cfg.get("video_dir", "")
            if video_dir_cfg:
                # demo 模式：使用 YAML 指定路径（与 _run_inference_for_model demo 分支一致）
                vd_path = Path(video_dir_cfg)
                if not vd_path.is_absolute():
                    vd_path = PROJECT_ROOT / vd_path
                vd_path = vd_path.resolve()
                if not vd_path.exists():
                    logger.warning(
                        f"[{model_cfg.get('name', model_key)}] --skip_inference: "
                        f"video_dir 指定目录不存在：{vd_path}"
                    )
                model_video_dirs[model_key] = vd_path
            else:
                # full pipeline 输出目录
                video_dir = output_dir / model_key
                if not video_dir.exists():
                    logger.warning(
                        f"跳过推理，但模型 '{model_key}' 的视频目录不存在：{video_dir}"
                    )
                model_video_dirs[model_key] = video_dir
    else:
        logger.info("=" * 60)
        logger.info("开始批量推理阶段")
        logger.info("=" * 60)
        for model_key in available_models:
            model_cfg = model_configs[model_key]
            video_dir = _run_inference_for_model(
                model_key=model_key,
                model_cfg=model_cfg,
                test_pairs=test_pairs,
                output_dir=output_dir,
                args=args,
                gpu_ids=gpu_ids,
            )
            # video_dir 可能为 None（demo 模式但目录不存在）
            model_video_dirs[model_key] = video_dir

    # ---- Step 3：VBench 评分 ----
    all_scores: Dict[str, Dict[str, Optional[float]]] = {}
    all_per_clip: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

    if not args.skip_vbench:
        logger.info("=" * 60)
        logger.info("开始 VBench 评分阶段（并发，共 {} GPU）".format(len(gpu_ids)))
        logger.info("=" * 60)
        _check_vbench_installed()

        # 初始化 all_scores / all_per_clip 结构
        for model_key in available_models:
            all_scores[model_key] = {}
            all_per_clip[model_key] = {}

        # 构建任务列表：跳过视频目录不存在的模型
        valid_tasks = []
        for model_key in available_models:
            model_name = model_configs[model_key].get("name", model_key)
            video_dir = model_video_dirs.get(model_key)
            if video_dir is None or not video_dir.exists():
                logger.warning(
                    f"模型 '{model_name}' 的视频目录不存在或为 None，跳过 VBench 评分"
                    + (f"：{video_dir}" if video_dir else "")
                )
                for dim in args.dimensions:
                    all_scores[model_key][dim] = None
                    all_per_clip[model_key][dim] = {}
                continue
            for dim in args.dimensions:
                valid_tasks.append((model_key, dim, video_dir))

        # GPU slot 池
        gpu_pool: Queue = Queue()
        for gid in gpu_ids:
            gpu_pool.put(str(gid))

        def _run_task(model_key, dim, video_dir):
            model_name = model_configs[model_key].get("name", model_key)
            gpu_id = gpu_pool.get()
            try:
                score, clip_scores = _run_vbench_single_dim(
                    model_key, model_name, video_dir, dim,
                    args.vbench_mode, output_dir, gpu_id
                )
                return model_key, dim, score, clip_scores
            finally:
                gpu_pool.put(gpu_id)

        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {
                executor.submit(_run_task, mk, d, vd): (mk, d)
                for (mk, d, vd) in valid_tasks
            }
            for fut in as_completed(futures):
                mk, d = futures[fut]
                try:
                    mk, d, score, clip_scores = fut.result()
                except Exception as e:
                    logger.warning(f"[{mk}] dim={d} 评测异常：{e}，结果置 None")
                    score, clip_scores = None, {}
                all_scores[mk][d] = score
                all_per_clip[mk][d] = clip_scores
    else:
        logger.info("--skip_vbench 已设置，跳过 VBench 评分阶段。")

    # ---- Step 4：汇总结果 ----
    if all_scores:
        logger.info("=" * 60)
        logger.info("汇总评测结果")
        logger.info("=" * 60)

        # 4a: aggregate CSV + Markdown 打印
        _summarize_results(
            all_scores=all_scores,
            model_configs=model_configs,
            dimensions=args.dimensions,
            output_dir=output_dir,
        )

        # 4b: per-clip CSV（若有数据）
        _summarize_per_clip_results(
            all_per_clip=all_per_clip,
            model_configs=model_configs,
            dimensions=args.dimensions,
            output_dir=output_dir,
            model_video_dirs=model_video_dirs,
        )

        # ---- 4c：追加到跨 run 注册表 all_runs.csv ----
        new_rows = []
        for mk in available_models:
            for dim in args.dimensions:
                clip_scores = all_per_clip.get(mk, {}).get(dim, {})
                if clip_scores:
                    for clip_id, score in clip_scores.items():
                        new_rows.append({
                            "run_name": run_name,
                            "clip_id": clip_id,
                            "dimension": dim,
                            "score": "" if score is None else f"{score:.6f}",
                        })
                else:
                    # 无 per-clip，仅写 aggregate
                    agg_score = (all_scores.get(mk) or {}).get(dim)
                    new_rows.append({
                        "run_name": run_name,
                        "clip_id": "_aggregate",
                        "dimension": dim,
                        "score": "" if agg_score is None else f"{agg_score:.6f}",
                    })

        if new_rows:
            write_header = not all_runs_csv.exists()
            with open(all_runs_csv, "a", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["run_name", "clip_id", "dimension", "score"]
                )
                if write_header:
                    writer.writeheader()
                writer.writerows(new_rows)
            logger.info(f"已追加 {len(new_rows)} 行到 {all_runs_csv}")
            _update_comparison_files(comparison_dir, all_runs_csv)
    else:
        logger.info("无评分数据可汇总（仅执行了推理或全部跳过）。")

    logger.info("eval_vbench.py 全部完成。")


if __name__ == "__main__":
    main()
