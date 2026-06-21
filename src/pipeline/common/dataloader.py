"""
dataloader.py — CSGO 数据集加载器（LingBot-World Memory Enhancement）

提供：
  - CSGODataset: 读取预处理好的 CSGO clip 目录（poses/actions/intrinsics + 路径）
  - compute_plucker_rays: 独立函数，将 c2w 矩阵 + intrinsics → Plucker 嵌入 [N, 6, H, W]
  - build_dit_cond_dict: 独立函数，构建 dit_cond_dict 中的 "c2ws_plucker_emb" 控制信号
  - collate_fn: DataLoader collate 函数，支持懒加载 VAE encode（默认跳过）
  - build_dataloader: 工厂函数，构建 DataLoader

接口约定：
  - __getitem__ 只返回路径 + numpy 数据，VAE encode 由 train.py 在 training_step 中完成
  - dit_cond_dict["c2ws_plucker_emb"] shape: tuple of [1, C_folded, lat_f, lat_h, lat_w]
    其中 C_folded = control_dim * (h/lat_h) * (w/lat_w) = 7 * 8 * 8 = 448
    格式与 train_lingbot_csgo.py prepare_control_signal + WanModel.forward() 完全一致：
      - WanModel.forward() 内部再做 patch_size=(1,2,2) 的空间折叠
      - 最终输入 patch_embedding_wancamctrl: Linear(7*64*4=1792, dim)

坐标系：
  - c2w 矩阵为 OpenCV 约定（x=右, y=下, z=前），由 preprocess_csgo_v3.py 已完成转换
  - intrinsics [fx, fy, cx, cy] 单位为像素（非归一化）

参考文件：
  - refs/lingbot-csgo-finetune/train_lingbot_csgo.py   prepare_control_signal（完整逻辑）
  - refs/lingbot-world/wan/utils/cam_utils.py          interpolate_camera_poses, compute_relative_poses,
                                                        get_plucker_embeddings, get_Ks_transformed
  - refs/lingbot-world/wan/modules/model.py            WanModel.forward()：c2ws_plucker_emb 消费方式
  - src/memory_module/model_with_memory.py             get_projected_frame_embs：期望 shape 确认

修复记录（对应 ReviewAgent 报告）：
  B-01/B-02/B-04: build_dit_cond_dict 中加入 interpolate_camera_poses + compute_relative_poses
                  + actions[::stride] 下采样
  B-03: c2ws_plucker_emb 使用 pixel→channel folding 后返回 tuple[Tensor[1,C,lat_f,lat_h,lat_w]]，
        与 WanModel.forward() 期望格式完全一致
  W-03: collate_fn 中 vae is not None 时 raise NotImplementedError
"""

import csv
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path：引入 lingbot-world cam_utils
# 本文件位于 src/pipeline/common/，需要向上三层到 Lingbot_LSM/，再进入 refs/lingbot-world
# ---------------------------------------------------------------------------
_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))    # → src/pipeline/common/
_PIPELINE_DIR = os.path.dirname(_COMMON_DIR)               # → src/pipeline/
_SRC_DIR = os.path.dirname(_PIPELINE_DIR)                   # → src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                   # → Lingbot_LSM/
_LINGBOT_WORLD = os.path.join(_PROJECT_ROOT, 'refs', 'lingbot-world')
if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

# 延迟导入（运行时才真正 import，避免在没有 scipy 的环境中 import 本模块报错）
def _get_cam_utils():
    """返回 (interpolate_camera_poses, compute_relative_poses,
              get_plucker_embeddings, get_Ks_transformed) 四个函数。
    来源：refs/lingbot-world/wan/utils/cam_utils.py
    """
    from wan.utils.cam_utils import (
        interpolate_camera_poses,
        compute_relative_poses,
        get_plucker_embeddings,
        get_Ks_transformed,
    )
    return interpolate_camera_poses, compute_relative_poses, get_plucker_embeddings, get_Ks_transformed


# ---------------------------------------------------------------------------
# 独立函数：compute_plucker_rays（保留，供外部独立调用）
# ---------------------------------------------------------------------------

def compute_plucker_rays(
    c2ws: Tensor,
    intrinsics: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """将 camera-to-world 矩阵和相机内参转换为 Plucker 嵌入（6 通道）。

    注意：此函数使用本模块自己的实现（不依赖 cam_utils），供独立调用。
    在 build_dit_cond_dict 内部直接使用 get_plucker_embeddings（cam_utils 版本）。

    Args:
        c2ws:       [N, 4, 4] float32，camera-to-world 矩阵，OpenCV 坐标系
        intrinsics: [N, 4] float32，[fx, fy, cx, cy]，像素单位（非归一化）
        height:     图像高度（像素），例如 480
        width:      图像宽度（像素），例如 832

    Returns:
        plucker: [N, 6, height, width] float32
                 通道 0-2: ray_origin（各 pixel 相同，即相机位置）
                 通道 3-5: ray_direction（单位向量，每像素不同）
    """
    n_frames = c2ws.shape[0]
    device = c2ws.device
    dtype = c2ws.dtype

    x_range = torch.arange(width, device=device, dtype=dtype) + 0.5
    y_range = torch.arange(height, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
    grid_xy = grid_xy[None, ...].expand(n_frames, -1, -1)

    fx = intrinsics[:, 0:1]
    fy = intrinsics[:, 1:2]
    cx = intrinsics[:, 2:3]
    cy = intrinsics[:, 3:4]

    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    xs = (i - cx) / fx
    ys = (j - cy) / fy
    zs = torch.ones_like(xs)

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    R = c2ws[:, :3, :3]
    rays_d = directions @ R.transpose(-1, -2)

    rays_o = c2ws[:, :3, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d)

    plucker = torch.cat([rays_o, rays_d], dim=-1)
    plucker = plucker.view(n_frames, height, width, 6)
    plucker = plucker.permute(0, 3, 1, 2).contiguous()

    return plucker


# ---------------------------------------------------------------------------
# 独立函数：build_dit_cond_dict
# ---------------------------------------------------------------------------

def build_dit_cond_dict(
    poses: Tensor,
    actions: Tensor,
    intrinsics: Tensor,
    height: int = 480,
    width: int = 832,
    vae_stride: Tuple[int, int, int] = (4, 8, 8),
) -> Dict[str, tuple]:
    """根据 pose/action/intrinsics 构建 WanModelWithMemory 所需的控制信号字典。

    完整复现 train_lingbot_csgo.py prepare_control_signal 逻辑（B-01/B-02/B-03/B-04 修复）：

    Step 1: get_Ks_transformed — 将 intrinsics 从原始分辨率变换到 height×width
    Step 2: interpolate_camera_poses — 将 81 帧绝对姿态插值到 lat_f 帧
            tgt_indices = linspace(0, 80, lat_f)，与 prepare_control_signal 一致：
            lat_f = int((num_frames - 1) // vae_stride[0]) + 1
    Step 3: compute_relative_poses — 绝对姿态 → 相对姿态（framewise=True, normalize_trans=True）
    Step 4: actions[::vae_stride[0]] — 下采样动作序列到 lat_f 帧（对齐相机插值帧数）
    Step 5: get_plucker_embeddings — [lat_f, h, w, 3]（only_rays_d=True）
    Step 6: pixel→channel folding，与 WanModel.forward() 内的 patch_size folding 配合：
            c1 = h // lat_h, c2 = w // lat_w（vae stride 8 → 60×104）
            rearrange: 'f (h c1) (w c2) c -> (f h w) (c c1 c2)'
            then reshape to [1, C_folded, lat_f, lat_h, lat_w]
            C_folded = 3 * c1 * c2 = 3 * 8 * 8 = 192 (rays_d) / 4 * 64 = 256 (wasd) → 448 total
    Step 7: 返回 tuple([1, 448, lat_f, lat_h, lat_w]) 供 WanModel.forward() 直接消费
            WanModel 内部再做 patch_size=(1,2,2) folding → [1, L, 1792] → Linear → [1, L, dim]

    Args:
        poses:      [N, 4, 4] float32，camera-to-world 矩阵（81 帧原始绝对姿态）
        actions:    [N, 4] float32，WASD 动作（0/1）
        intrinsics: [N, 4] float32，[fx, fy, cx, cy]，像素单位
        height:     图像高度，默认 480（数据集原始高度）
        width:      图像宽度，默认 832（数据集原始宽度）
        vae_stride: VAE 时间×空间下采样倍率，默认 (4, 8, 8)

    Returns:
        dict with key "c2ws_plucker_emb": tuple of one Tensor [1, 448, lat_f, lat_h, lat_w]
        （tuple 格式与 WanModel.forward() 中 `for i in c2ws_plucker_emb` 一致）
    """
    (interpolate_camera_poses,
     compute_relative_poses,
     get_plucker_embeddings,
     get_Ks_transformed) = _get_cam_utils()

    num_frames = poses.shape[0]  # 81
    stride_t, stride_h, stride_w = vae_stride  # (4, 8, 8)

    # 计算 latent 维度
    lat_f = int((num_frames - 1) // stride_t) + 1   # (81-1)//4+1 = 21
    lat_h = height // stride_h                       # 480//8 = 60
    lat_w = width // stride_w                        # 832//8 = 104

    # pixel→channel 下采样倍率（vae 空间 stride）
    fold_h = stride_h   # = h // lat_h = 8
    fold_w = stride_w   # = w // lat_w = 8

    # ---- Step 1: 变换 intrinsics（处理分辨率缩放，此处 src=dst 故无缩放）----
    Ks = get_Ks_transformed(
        intrinsics,
        height_org=height, width_org=width,
        height_resize=height, width_resize=width,
        height_final=height, width_final=width,
    )  # [N, 4]，此处 height_org=height_resize=height_final 故 Ks 不变
    Ks_single = Ks[0]  # [4]，取第一帧（clip 内 intrinsics 固定）

    # ---- Step 2: 插值 poses 81帧 → lat_f 帧 ----
    src_indices = np.linspace(0, num_frames - 1, num_frames)          # [0, 1, ..., 80]
    tgt_indices = np.linspace(0, num_frames - 1, lat_f)               # 21 个等间距索引
    c2ws_infer = interpolate_camera_poses(
        src_indices=src_indices,
        src_rot_mat=poses[:, :3, :3].cpu().numpy(),
        src_trans_vec=poses[:, :3, 3].cpu().numpy(),
        tgt_indices=tgt_indices,
    )  # [lat_f, 4, 4]，torch.float32

    # ---- Step 3: 绝对姿态 → 相对姿态 ----
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)   # [lat_f, 4, 4]

    # ---- Step 4: actions 下采样 81 → lat_f 帧（与 camera 插值帧对齐）----
    # 与 prepare_control_signal 中 actions[::4] 完全一致（stride_t=4）
    # processed_csgo_v3 使用 8ch action，截断到前 4 维（WASD）以兼容 v2 模型架构
    wasd = actions[::stride_t, :4]  # [lat_f 或 >=lat_f, 4]
    # 长度对齐（截断或末帧填充）
    if len(wasd) > lat_f:
        wasd = wasd[:lat_f]
    elif len(wasd) < lat_f:
        pad = wasd[-1:].expand(lat_f - len(wasd), -1)
        wasd = torch.cat([wasd, pad], dim=0)
    # wasd: [lat_f, 4]

    # 同设备
    device = poses.device
    c2ws_infer = c2ws_infer.to(device)
    Ks_repeated = Ks_single.unsqueeze(0).expand(lat_f, -1).to(device)  # [lat_f, 4]
    wasd = wasd.to(device)

    # ---- Step 5: Plucker 嵌入（only_rays_d=True，3 通道）----
    # 返回 [lat_f, height, width, 3]
    rays_d_map = get_plucker_embeddings(
        c2ws_infer, Ks_repeated, height, width, only_rays_d=True
    )  # [lat_f, h, w, 3]

    # ---- Step 6a: rays_d pixel→channel folding ----
    # 'f (h c1) (w c2) c -> (f h w) (c c1 c2)'  c1=fold_h, c2=fold_w
    c2ws_plucker_emb = rearrange(
        rays_d_map,
        'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
        c1=fold_h, c2=fold_w,
    )  # [(lat_f * lat_h * lat_w), 3*fold_h*fold_w]
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [1, lat_f*lat_h*lat_w, 192]
    c2ws_plucker_emb = rearrange(
        c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
        f=lat_f, h=lat_h, w=lat_w,
    )  # [1, 192, lat_f, lat_h, lat_w]

    # ---- Step 6b: WASD 广播到全分辨率，再做相同 pixel→channel folding ----
    # wasd: [lat_f, 4] → broadcast to [lat_f, height, width, 4]
    wasd_map = wasd[:, None, None, :].expand(lat_f, height, width, -1)  # [lat_f, h, w, 4]
    wasd_tensor = rearrange(
        wasd_map,
        'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
        c1=fold_h, c2=fold_w,
    )  # [(lat_f * lat_h * lat_w), 4*fold_h*fold_w]
    wasd_tensor = wasd_tensor[None, ...]  # [1, lat_f*lat_h*lat_w, 256]
    wasd_tensor = rearrange(
        wasd_tensor, 'b (f h w) c -> b c f h w',
        f=lat_f, h=lat_h, w=lat_w,
    )  # [1, 256, lat_f, lat_h, lat_w]

    # ---- Step 7: 拼接 → [1, 448, lat_f, lat_h, lat_w]，转 bfloat16 ----
    c2ws_plucker_emb = torch.cat(
        [c2ws_plucker_emb, wasd_tensor], dim=1
    ).to(torch.bfloat16)
    # C_folded = 3*64 + 4*64 = 192 + 256 = 448
    # WanModel.forward() 内部再做 patch_size=(1,2,2) folding:
    #   448 * 1 * 2 * 2 = 1792 → patch_embedding_wancamctrl: Linear(1792, dim) ✓

    # 返回 tuple 格式：WanModel.forward() 中 `for i in c2ws_plucker_emb` 逐元素迭代
    return {"c2ws_plucker_emb": (c2ws_plucker_emb,)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CSGODataset(Dataset):
    """CSGO 预处理数据集（用于 LingBot-World Memory Enhancement 微调）。

    读取 metadata_{split}.csv，每条样本对应一个 clip 目录。
    __getitem__ 只返回文件路径和 numpy 数组，不做 VAE encode（encode 很慢，
    应在 training_step 中对整个 batch 做一次）。

    目录结构（每个 clip）：
        {dataset_root}/{clip_path}/
            video.mp4       16fps, 480×832, 81帧
            image.jpg       第一帧静态图
            poses.npy       [81, 4, 4] float32，camera-to-world 矩阵
            action.npy      [81, 4] int32，WASD 动作（0/1）
            intrinsics.npy  [81, 4] float32，[fx, fy, cx, cy]（clip 内固定）
            prompt.txt      文字描述

    Args:
        metadata_csv:  metadata CSV 文件的绝对路径
                       （通常为 {dataset_root}/metadata_train.csv）
        dataset_root:  数据集根目录，用于拼接 clip_path
        vae:           保留参数（当前未使用；encode 在 train.py 中进行）
        transform:     保留参数（当前未使用；视频只返回路径）
        max_frames:    clip 最大帧数（默认 81）；超长 clip 截断，短 clip 末帧填充
    """

    def __init__(
        self,
        metadata_csv: str,
        dataset_root: str,
        vae=None,
        transform=None,
        max_frames: int = 81,
    ):
        self.dataset_root = dataset_root
        self.vae = vae          # 保留，当前不使用
        self.transform = transform  # 保留，当前不使用
        self.max_frames = max_frames

        # ---- 读取 CSV ----
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"metadata CSV 不存在: {metadata_csv}")

        self.samples: List[Dict[str, str]] = []
        with open(metadata_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(dict(row))

        if len(self.samples) == 0:
            raise ValueError(f"metadata CSV 中无有效样本: {metadata_csv}")

        logger.info(
            "CSGODataset: 加载 %d 个样本，metadata=%s，root=%s",
            len(self.samples), metadata_csv, dataset_root,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """返回第 idx 个 clip 的数据字典。

        Returns:
            dict with keys:
              "video_path":   str，video.mp4 的绝对路径（供 train.py VAE encode）
              "image_path":   str，image.jpg 的绝对路径（供训练时 encode 第一帧条件）
              "poses":        Tensor [max_frames, 4, 4] float32，camera-to-world 矩阵
              "actions":      Tensor [max_frames, 4] float32（int32 → float32）
              "intrinsics":   Tensor [max_frames, 4] float32，[fx, fy, cx, cy]
              "prompt":       str，文字描述（优先读 prompt.txt，fallback 到 CSV）
              "clip_path":    str，clip 目录的相对路径（来自 CSV）
        """
        sample = self.samples[idx]
        clip_rel = sample["clip_path"]
        clip_dir = os.path.join(self.dataset_root, clip_rel)

        try:
            # ---- 路径 ----
            video_path = os.path.join(clip_dir, "video.mp4")
            image_path = os.path.join(clip_dir, "image.jpg")

            # ---- 加载 numpy 数组 ----
            poses = np.load(os.path.join(clip_dir, "poses.npy"))         # [F, 4, 4] float32
            actions = np.load(os.path.join(clip_dir, "action.npy"))      # [F, 4] int32
            intrinsics = np.load(os.path.join(clip_dir, "intrinsics.npy"))  # [F, 4] float32

            # ---- 帧数对齐（截断或末帧填充）----
            poses = self._pad_or_truncate(poses, self.max_frames)
            actions = self._pad_or_truncate(actions, self.max_frames)
            intrinsics = self._pad_or_truncate(intrinsics, self.max_frames)

            # ---- Tensor 转换 ----
            poses_t = torch.from_numpy(poses).float()       # [max_frames, 4, 4]
            actions_t = torch.from_numpy(actions).float()   # [max_frames, 4]（int→float）
            intrinsics_t = torch.from_numpy(intrinsics).float()  # [max_frames, 4]

            # ---- Prompt（优先读文件，fallback 到 CSV）----
            prompt_file = os.path.join(clip_dir, "prompt.txt")
            if os.path.exists(prompt_file):
                with open(prompt_file, "r") as pf:
                    prompt = pf.read().strip()
            else:
                prompt = sample.get("prompt", "")
                logger.warning("prompt.txt 不存在，使用 CSV 字段: %s", clip_dir)

        except Exception as e:
            # ---- 异常处理：跳过损坏样本，返回最近合法样本 ----
            # 数据集规范（code_standards.md §3）：__getitem__ 不能崩溃
            logger.error(
                "加载样本失败 (idx=%d, clip=%s): %s，返回 idx=0 作为替代",
                idx, clip_rel, e,
            )
            if idx != 0:
                return self.__getitem__(0)
            # idx==0 也失败时抛出，避免死循环
            raise

        return {
            "video_path": video_path,
            "image_path": image_path,
            "poses": poses_t,
            "actions": actions_t,
            "intrinsics": intrinsics_t,
            "prompt": prompt,
            "clip_path": clip_rel,
        }

    @staticmethod
    def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
        """将数组截断到 target_len 或用末行填充到 target_len。"""
        if len(arr) >= target_len:
            return arr[:target_len]
        rep_shape = (target_len - len(arr),) + (1,) * (arr.ndim - 1)
        pad = np.tile(arr[-1:], rep_shape)
        return np.concatenate([arr, pad], axis=0)


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Dict],
    vae=None,
    height: int = 480,
    width: int = 832,
    vae_stride: Tuple[int, int, int] = (4, 8, 8),
) -> Dict:
    """DataLoader collate 函数，将 CSGODataset.__getitem__ 列表合并为 batch dict。

    VAE encode 策略：
      - vae=None（默认）：latents 返回 None，由 train.py 自行 encode 整个 batch
      - vae 非 None：抛出 NotImplementedError（W-03 修复：不允许在 collate_fn 中静默跳过）

    Args:
        batch:       list of dicts，来自 CSGODataset.__getitem__
        vae:         可选 VAE 模型（必须为 None；非 None 时抛出异常）
        height:      视频高度（默认 480）
        width:       视频宽度（默认 832）
        vae_stride:  VAE 下采样倍率（默认 (4, 8, 8)），传入 build_dit_cond_dict

    Returns:
        dict with keys:
          "latents":       None（VAE encode 由 train.py 在 training_step 中完成）
          "dit_cond_dict": dict, "c2ws_plucker_emb": list of tuples，
                           每个 tuple 含 one Tensor [1, 448, lat_f, lat_h, lat_w]
          "prompts":       list[str]
          "first_frames":  list[str]，image.jpg 路径（train.py encode 第一帧条件）
          "video_paths":   list[str]，video.mp4 路径（供 train.py VAE encode）
          "clip_paths":    list[str]，clip 相对路径（日志/调试用）
    """
    # W-03: vae 非 None 时明确报错，不允许静默跳过
    if vae is not None:
        raise NotImplementedError(
            "collate_fn 不支持在 DataLoader worker 中 VAE encode 视频。"
            "请在 train.py 的 training_step 中自行 encode，并保持 vae=None。"
        )

    prompts = [item["prompt"] for item in batch]
    first_frames = [item["image_path"] for item in batch]
    video_paths = [item["video_path"] for item in batch]
    clip_paths = [item["clip_path"] for item in batch]

    # ---- Stack tensors ----
    poses_batch = torch.stack([item["poses"] for item in batch], dim=0)         # [B, T, 4, 4]
    actions_batch = torch.stack([item["actions"] for item in batch], dim=0)     # [B, T, 4]
    intrinsics_batch = torch.stack([item["intrinsics"] for item in batch], dim=0)  # [B, T, 4]

    B = poses_batch.shape[0]

    # ---- 构建 dit_cond_dict（逐样本计算后组成 list）----
    # 每个样本独立计算以避免在 collate 阶段做大批量矩阵乘法
    # 训练循环中通常 batch_size=1（与 train_lingbot_csgo.py 一致），
    # 多样本时 dit_cond_dict["c2ws_plucker_emb"] 是一个 list of tuples。
    all_plucker_tuples = []
    for b in range(B):
        cond = build_dit_cond_dict(
            poses=poses_batch[b],           # [T, 4, 4]
            actions=actions_batch[b],       # [T, 4]
            intrinsics=intrinsics_batch[b], # [T, 4]
            height=height,
            width=width,
            vae_stride=vae_stride,
        )
        # cond["c2ws_plucker_emb"] 是 tuple of one Tensor [1, 448, lat_f, lat_h, lat_w]
        all_plucker_tuples.append(cond["c2ws_plucker_emb"][0])  # Tensor [1, 448, lat_f, lat_h, lat_w]

    # 以 tuple 形式传入，保持与 WanModel.forward() `for i in c2ws_plucker_emb` 兼容
    dit_cond_dict = {
        "c2ws_plucker_emb": tuple(all_plucker_tuples),
    }

    return {
        "latents": None,
        "dit_cond_dict": dit_cond_dict,
        "prompts": prompts,
        "first_frames": first_frames,
        "video_paths": video_paths,
        "clip_paths": clip_paths,
    }


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_dataloader(
    metadata_csv: str,
    dataset_root: str,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    vae=None,
    height: int = 480,
    width: int = 832,
    max_frames: int = 81,
    vae_stride: Tuple[int, int, int] = (4, 8, 8),
    **kwargs,
) -> DataLoader:
    """构建 CSGODataset + DataLoader。

    DataLoader collate 函数已内置，使用闭包绑定 vae/height/width/vae_stride 参数。

    Args:
        metadata_csv:  metadata CSV 绝对路径（如 /data/metadata_train.csv）
        dataset_root:  数据集根目录（clip_path 相对此目录）
        batch_size:    每个 batch 的样本数
        num_workers:   DataLoader 工作进程数（默认 4，满足 code_standards.md §3）
        shuffle:       是否打乱数据（训练时 True，验证时 False）
        vae:           必须为 None（collate_fn 不支持 VAE encode，见 W-03）
        height:        视频高度（默认 480）
        width:         视频宽度（默认 832）
        max_frames:    clip 最大帧数（默认 81）
        vae_stride:    VAE 时间×空间下采样倍率（默认 (4, 8, 8)）
        **kwargs:      其他传给 DataLoader 的参数（如 pin_memory, drop_last 等）

    Returns:
        torch.utils.data.DataLoader
    """
    if vae is not None:
        raise NotImplementedError(
            "build_dataloader 不支持在 DataLoader 中 VAE encode。"
            "请保持 vae=None，在 training_step 中自行 encode。"
        )

    dataset = CSGODataset(
        metadata_csv=metadata_csv,
        dataset_root=dataset_root,
        vae=None,
        max_frames=max_frames,
    )

    # 闭包绑定 collate 参数
    def _collate(batch_list: List[Dict]) -> Dict:
        return collate_fn(
            batch_list,
            vae=None,
            height=height,
            width=width,
            vae_stride=vae_stride,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        **kwargs,
    )

    logger.info(
        "build_dataloader: %d 样本，batch_size=%d，num_workers=%d，shuffle=%s",
        len(dataset), batch_size, num_workers, shuffle,
    )
    return loader
