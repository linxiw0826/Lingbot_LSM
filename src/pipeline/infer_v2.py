"""
infer_v2.py — CSGO 推理脚本，完全对齐 csgo-finetune-v2/inference_csgo.py，
             在其基础上支持 WanModelWithMemory + Memory Bank。

与 inference_csgo.py 的关系：
  - 保留所有原有参数与行为（LoRA 加载、全参微调加载、WanI2V 推理）
  - 新增 --use_memory / --memory_max_size，控制是否启用 Memory Bank
  - 当 --use_memory False 时，与 inference_csgo.py 行为完全一致

用法（无 memory，等价于 inference_csgo.py）：
    torchrun --nproc_per_node=8 infer_v2.py \\
        --ckpt_dir /path/to/lingbot-world-base-act/ \\
        --lora_path /path/to/lora_weights.pth \\
        --image /path/to/image.jpg \\
        --action_path /path/to/clip/ \\
        --prompt "First-person view of CS:GO gameplay" \\
        --size 480*832 --frame_num 81

用法（启用 memory）：
    torchrun --nproc_per_node=8 infer_v2.py \\
        --ckpt_dir /path/to/lingbot-world-base-act/ \\
        --lora_path /path/to/lora_weights.pth \\
        --image /path/to/image.jpg \\
        --action_path /path/to/clip/ \\
        --prompt "First-person view of CS:GO gameplay" \\
        --size 480*832 --frame_num 81 \\
        --use_memory --memory_max_size 50

用法（启用 memory + dual 模型，对应 train_v2_stage1_dual.py 输出）：
    torchrun --nproc_per_node=4 infer_v2.py \\
        --ckpt_dir /path/to/lingbot-world-base-act/ \\
        --ft_model_dir /path/to/output/low_noise_model/ \\
        --ft_high_model_dir /path/to/output/high_noise_model/ \\
        --image /path/to/image.jpg \\
        --action_path /path/to/clip/ \\
        --prompt "First-person view of CS:GO competitive gameplay" \\
        --size 480*832 --frame_num 81 \\
        --use_memory --memory_max_size 50 --num_clips 5
"""

import argparse
import logging
import os
import sys
from os.path import abspath, dirname, join

import torch

# ---------------------------------------------------------------------------
# sys.path 设置
# ---------------------------------------------------------------------------

_PIPELINE_DIR = dirname(abspath(__file__))          # → src/pipeline/
_SRC_DIR = dirname(_PIPELINE_DIR)                   # → src/
_PROJECT_ROOT = dirname(_SRC_DIR)                   # → Lingbot_LSM/
_LINGBOT_WORLD = join(_PROJECT_ROOT, 'refs', 'lingbot-world')

if _LINGBOT_WORLD not in sys.path:
    sys.path.insert(0, _LINGBOT_WORLD)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ulysses SP + Memory 联合推理支持
# ---------------------------------------------------------------------------

# 用于 SP 模式下在模型属性上传递 memory states（替代 forward 替换）
_SP_MEM_STATES_ATTR = '_sp_memory_states'
_SP_MEM_VAL_ATTR    = '_sp_memory_value_states'


def _sp_dit_forward_with_memory(self, x, t, context, seq_len, y=None, dit_cond_dict=None):
    """WanModelWithMemory + Ulysses SP 联合 forward。

    读取 self._sp_memory_states / self._sp_memory_value_states（由 _patch_pipeline_memory 设置），
    将其注入 dit_cond_dict，然后按 sp_dit_forward 的逻辑进行序列并行推理。

    与原版 sp_dit_forward 的区别：先注入 memory，再做 SP 分割和 block 调用。
    blocks 是 MemoryBlockWrapper，其内层 self_attn.forward 已被 sp_attn_forward 替换。
    """
    import torch.nn.functional as torch_F
    from wan.distributed.util import get_rank, get_world_size, gather_forward
    from wan.modules.model import sinusoidal_embedding_1d
    from einops import rearrange
    # 延迟导入，避免循环依赖
    from memory_module.model_with_memory import _MEMORY_STATES_KEY, _MEMORY_VALUE_KEY

    # ---- Step 1: 注入 memory states ----
    _mem_states = getattr(self, _SP_MEM_STATES_ATTR, None)
    _mem_val    = getattr(self, _SP_MEM_VAL_ATTR,    None)
    if _mem_states is not None:
        if dit_cond_dict is None:
            dit_cond_dict = {}
        else:
            dit_cond_dict = dict(dit_cond_dict)
        _dev = self.patch_embedding.weight.device
        dit_cond_dict[_MEMORY_STATES_KEY] = _mem_states.to(_dev)
        if _mem_val is not None:
            dit_cond_dict[_MEMORY_VALUE_KEY] = _mem_val.to(_dev)

    # ---- Step 2: sp_dit_forward 逻辑（与原版完全一致）----
    # Ulysses SP 对齐保护：seq_len 必须是 world_size 的整数倍，否则 chunk 不均匀
    # 导致 rope_apply pad_freqs 收到负 pad_size（见 bug 2026-05-06）
    _ws = get_world_size()
    if _ws > 1 and seq_len % _ws != 0:
        seq_len = (_ws - seq_len % _ws) + seq_len

    if self.model_type == 'i2v':
        assert y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
        c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
        c2ws_plucker_emb = [
            rearrange(
                i, '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                c1=self.patch_size[0], c2=self.patch_size[1], c3=self.patch_size[2],
            ) for i in c2ws_plucker_emb
        ]
        c2ws_plucker_emb = torch.cat(c2ws_plucker_emb, dim=1)
        c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
        c2ws_hidden = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
        c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden
        cam_len = c2ws_plucker_emb.size(1)
        if cam_len < seq_len:
            pad = c2ws_plucker_emb.new_zeros(
                c2ws_plucker_emb.size(0), seq_len - cam_len, c2ws_plucker_emb.size(2))
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, pad], dim=1)
        elif cam_len > seq_len:
            c2ws_plucker_emb = c2ws_plucker_emb[:, :seq_len, :]
        if get_world_size() > 1:
            c2ws_plucker_emb = torch.chunk(c2ws_plucker_emb, get_world_size(), dim=1)[get_rank()]
        dit_cond_dict = dict(dit_cond_dict)
        dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb

    x  = torch.chunk(x,  get_world_size(), dim=1)[get_rank()]
    e  = torch.chunk(e,  get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
        context=context, context_lens=context_lens, dit_cond_dict=dit_cond_dict)

    for block in self.blocks:
        x = block(x, **kwargs)

    x = self.head(x, e)
    x = gather_forward(x, dim=1)
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def _configure_memory_model_for_dist(model, use_sp: bool, device):
    """为 WanModelWithMemory 应用 Ulysses SP（不使用 FSDP，权重在各卡复制）。

    必须在 _convert_pipeline_to_memory 之后调用，此时 model.blocks 已是
    MemoryBlockWrapper 列表，内层 self_attn 在 block.block.self_attn。

    Args:
        model:   WanModelWithMemory 实例
        use_sp:  是否启用 Ulysses 序列并行
        device:  目标设备（cuda:local_rank）
    """
    import types
    from memory_module.model_with_memory import MemoryBlockWrapper
    from wan.distributed.sequence_parallel import sp_attn_forward

    model.eval().requires_grad_(False)

    if use_sp:
        # 为每个 MemoryBlockWrapper 内部 block 的 self_attn 打补丁
        for block in model.blocks:
            if isinstance(block, MemoryBlockWrapper):
                inner_attn = block.block.self_attn
            else:
                inner_attn = block.self_attn  # 兼容未被 wrap 的普通 block
            inner_attn.forward = types.MethodType(sp_attn_forward, inner_attn)

        # 替换 model.forward 为 SP + memory 联合版本
        model.forward = types.MethodType(_sp_dit_forward_with_memory, model)

        # 初始化 memory state 属性（_patch_pipeline_memory 会按 clip 覆盖）
        setattr(model, _SP_MEM_STATES_ATTR, None)
        setattr(model, _SP_MEM_VAL_ATTR,    None)

    model.to(device)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="CSGO 推理脚本（完全对齐 inference_csgo.py + Memory Bank 扩展）"
    )

    # ---- v2 原有参数（与 inference_csgo.py 完全一致）----
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="基础模型目录（lingbot-world checkpoint）")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA 权重 .pth 文件路径（可选）")
    parser.add_argument("--ft_model_dir", type=str, default=None,
                        help="全参微调 low_noise_model 目录（可选，与 --lora_path 互斥）")
    parser.add_argument("--ft_high_model_dir", type=str, default=None,
                        help="dual 训练中 high_noise_model 输出目录（可选；"
                             "train_v2_stage1_dual.py 输出的 OUTPUT_DIR/high_noise_model/）。"
                             "提供时同时对 high_noise_model 启用 WanModelWithMemory。")
    parser.add_argument("--image", type=str, required=True,
                        help="初始帧图像路径")
    parser.add_argument("--action_path", type=str, default=None,
                        help="动作数据路径（action.npy 或含 poses.npy 的目录）")
    parser.add_argument("--prompt", type=str,
                        default="First-person view of CS:GO competitive gameplay",
                        help="文本描述")
    parser.add_argument("--save_file", type=str, default="output_csgo.mp4",
                        help="输出视频路径")
    parser.add_argument("--size", type=str, default="480*832",
                        help="分辨率，如 '480*832'")
    parser.add_argument("--frame_num", type=int, default=81,
                        help="帧数（默认 81）")
    parser.add_argument("--sample_steps", type=int, default=70,
                        help="采样步数（默认 70）")
    parser.add_argument("--sample_shift", type=float, default=10.0,
                        help="sigma shift（默认 10.0）")
    parser.add_argument("--guide_scale", type=float, default=5.0,
                        help="CFG scale（默认 5.0）")
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)

    # ---- Memory Enhancement 参数（新增）----
    parser.add_argument("--use_memory", action="store_true", default=False,
                        help="是否启用 Memory Bank（默认 False，保持后向兼容）")
    parser.add_argument("--memory_max_size", type=int, default=50,
                        help="Memory Bank 最大容量（默认 50）")
    parser.add_argument("--num_clips", type=int, default=1,
                        help="Memory 模式下生成的 clip 数量（默认 1，等价于原行为）")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Memory 模块 key 识别（LoRA/HF checkpoint 权重提取共用）
# ---------------------------------------------------------------------------

_MEMORY_KEY_PATTERNS = ("memory_cross_attn", "memory_norm", "nfp_head", "latent_proj")


# ---------------------------------------------------------------------------
# LoRA 加载（与 inference_csgo.py 完全一致）
# ---------------------------------------------------------------------------

def _load_lora_and_prepare_ckpt(args) -> str:
    """加载 LoRA 权重并合并，返回合并后的临时 ckpt_dir。

    流程与 inference_csgo.py 完全一致：
      1. 加载 WanModel（control_type='act'）
      2. 从 lora_state_dict 自动检测 target_modules 和 lora_rank
      3. inject_adapter_in_model(LoraConfig(...), model)
      4. 键名映射：lora_A.weight → lora_A.default.weight
      5. 合并 LoRA 权重：module.merge()
      6. model.save_pretrained(tmp_dir/low_noise_model)
      7. 符号链接其他文件（high_noise_model, VAE, T5 等）
      8. 返回 tmp_dir
    """
    logger.info("Loading base model + LoRA weights for inference...")

    from wan.modules.model import WanModel
    from peft import LoraConfig, inject_adapter_in_model

    # Step 1：加载 base low_noise_model（control_type='act'，control_dim=7）
    model = WanModel.from_pretrained(
        args.ckpt_dir, subfolder="low_noise_model",
        torch_dtype=torch.bfloat16, control_type="act",
    )

    # Step 2：从 lora_state_dict 自动检测 target_modules 和 lora_rank
    lora_state = torch.load(args.lora_path, map_location="cpu")
    target_modules = set()
    for key in lora_state.keys():
        # 提取模块名，例如 "blocks.0.self_attn.q.lora_A.default.weight" → "blocks.0.self_attn.q"
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part in ("lora_A", "lora_B"):
                module_name = ".".join(parts[:i])
                target_modules.add(module_name)
                break

    target_modules = sorted(list(target_modules))
    logger.info("Detected %d LoRA target modules", len(target_modules))

    # 自动检测 lora_rank（从 lora_A.shape[0]）
    lora_rank = None
    for key, val in lora_state.items():
        if "lora_A" in key:
            lora_rank = val.shape[0]
            break
    if lora_rank is None:
        raise ValueError("Cannot detect lora_rank from lora_state_dict — no lora_A key found.")
    logger.info("Detected lora_rank=%d", lora_rank)

    # Step 3：inject_adapter_in_model
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=target_modules)
    model = inject_adapter_in_model(lora_config, model)

    # Step 4：键名映射（lora_A.weight → lora_A.default.weight）
    mapped_state = {}
    for key, val in lora_state.items():
        if "lora_A.weight" in key and "default" not in key:
            key = key.replace("lora_A.weight", "lora_A.default.weight")
        if "lora_B.weight" in key and "default" not in key:
            key = key.replace("lora_B.weight", "lora_B.default.weight")
        mapped_state[key] = val

    result = model.load_state_dict(mapped_state, strict=False)
    logger.info(
        "Loaded LoRA weights: %d keys, missing=%d, unexpected=%d",
        len(mapped_state), len(result.missing_keys), len(result.unexpected_keys),
    )

    # Step 5：合并 LoRA 权重（module.merge()）
    import peft.tuners.lora as lora_module
    for _name, _module in model.named_modules():
        if isinstance(_module, lora_module.Linear):
            _module.merge()

    # Step 6：保存合并后的模型到临时目录
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='act_')
    merged_ckpt = os.path.join(tmp_dir, "low_noise_model")
    model.save_pretrained(merged_ckpt)
    logger.info("Saved merged model to %s", merged_ckpt)

    # Step 7：符号链接其他文件
    for item in ["high_noise_model", "Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                 "google", "configuration.json"]:
        src = os.path.join(args.ckpt_dir, item)
        dst = os.path.join(tmp_dir, item)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # Step 6b：保存 memory module 权重（V5-B2-01 fix）
    # lora_weights.pth 同时含有 LoRA adapter 权重 + memory 模块权重（gate/nfp_head/latent_proj）。
    # WanModel 不认识 memory 相关 key → load_state_dict 时被丢弃；需单独保存后由
    # _convert_pipeline_to_memory 加载，确保训练好的 gate/nfp_head/latent_proj 在推理时生效。
    memory_weights = {
        k: v for k, v in lora_state.items()
        if any(pat in k for pat in _MEMORY_KEY_PATTERNS)
    }
    if memory_weights:
        _mem_path = os.path.join(tmp_dir, "memory_weights.pth")
        torch.save(memory_weights, _mem_path)
        logger.info(
            "V5-B2-01 fix: saved %d memory module weights to %s",
            len(memory_weights), _mem_path,
        )

    del model, lora_state, mapped_state
    torch.cuda.empty_cache()

    return tmp_dir


def _load_ft_model_and_prepare_ckpt(args) -> str:
    """全参微调模型：准备 tmp_dir，WanI2V 始终从基础模型加载（避免 .block. key 不匹配警告）。

    说明：
      FT checkpoint 由 WanModelWithMemory 训练生成，所有 key 带 .block. 前缀。
      若直接将 FT 目录 symlink 给 WanI2V，WanModel.from_pretrained 会因 key 不匹配
      丢弃全部权重并打印大量 "Some weights not used" 警告。

      修复方案：tmp_dir/low_noise_model 和 tmp_dir/high_noise_model 始终指向基础模型，
      WanI2V 加载干净的基础权重；FT 权重由 _convert_pipeline_to_memory（Step 4）在
      转换为 WanModelWithMemory 后通过 load_state_dict(strict=False) 正确注入。
    """
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='act_')

    # low_noise_model：始终使用基础模型（FT 权重由 _convert_pipeline_to_memory 注入）
    _base_low = os.path.join(args.ckpt_dir, "low_noise_model")
    os.symlink(_base_low, os.path.join(tmp_dir, "low_noise_model"))

    # high_noise_model：始终使用基础模型（FT 权重同样由 _convert_pipeline_to_memory 注入）
    _base_high = os.path.join(args.ckpt_dir, "high_noise_model")
    _high_dst = os.path.join(tmp_dir, "high_noise_model")
    if os.path.exists(_base_high) and not os.path.exists(_high_dst):
        os.symlink(_base_high, _high_dst)

    for item in ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
                 "google", "configuration.json"]:
        src = os.path.join(args.ckpt_dir, item)
        dst = os.path.join(tmp_dir, item)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    return tmp_dir


# ---------------------------------------------------------------------------
# Memory 辅助（复用 infer.py 中的逻辑）
# ---------------------------------------------------------------------------

def _load_all_weights_from_hf_checkpoint(model_dir: str) -> dict:
    """从 checkpoint 目录中加载所有权重。

    按优先级依次尝试：
      1. *.safetensors（新版 HF 格式）
      2. diffusion_pytorch_model*.bin（ZeRO-3 save_16bit_model 输出，训练脚本默认格式）
      3. pytorch_model*.bin（旧版 HF 格式）
    """
    import glob
    state = {}

    # 1. safetensors
    sf_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if sf_files:
        try:
            from safetensors.torch import load_file
            for f in sf_files:
                state.update(load_file(f, device="cpu"))
        except ImportError:
            logger.warning("safetensors not available, skipping .safetensors for %s", model_dir)
            sf_files = []

    # 2. diffusion_pytorch_model*.bin（ZeRO-3 / save_16bit_model 格式）
    if not state:
        bin_files = sorted(glob.glob(os.path.join(model_dir, "diffusion_pytorch_model*.bin")))
        for f in bin_files:
            state.update(torch.load(f, map_location="cpu", weights_only=True))

    # 3. pytorch_model*.bin（旧版 HF 格式）
    if not state:
        bin_files = sorted(glob.glob(os.path.join(model_dir, "pytorch_model*.bin")))
        for f in bin_files:
            state.update(torch.load(f, map_location="cpu", weights_only=True))

    logger.info(
        "_load_all_weights_from_hf_checkpoint: loaded %d keys from %s",
        len(state), model_dir,
    )
    return state


def _load_memory_weights_from_hf_checkpoint(model_dir: str) -> dict:
    """从 HuggingFace 格式 checkpoint 中提取 memory module 权重（已废弃，内部改用 _load_all_weights）。

    保留此函数以兼容外部调用。
    """
    state = _load_all_weights_from_hf_checkpoint(model_dir)
    memory_state = {k: v for k, v in state.items()
                    if any(pat in k for pat in _MEMORY_KEY_PATTERNS)}
    logger.info(
        "_load_memory_weights_from_hf_checkpoint: found %d memory keys in %s",
        len(memory_state), model_dir,
    )
    return memory_state


def _convert_pipeline_to_memory(pipeline, memory_max_size: int, memory_ckpt_path=None,
                                 high_model_dir=None, low_model_dir=None):
    """将 WanI2V 管道的 low_noise_model 替换为 WanModelWithMemory，并加载微调权重。

    背景：WanModelWithMemory 将 WanAttentionBlock 包裹在 MemoryBlockWrapper.block 下，
    导致所有 checkpoint key 带有 .block. 前缀（如 blocks.N.block.self_attn.q.weight），
    与 WanModel.from_pretrained 期待的 blocks.N.self_attn.q.weight 不匹配。
    因此 WanI2V 初始化时会丢弃所有微调权重（含 cam_injector、memory 等）。
    本函数在转换为 WanModelWithMemory 后（key 结构与 checkpoint 完全吻合），
    通过 load_state_dict(strict=False) 重新加载全量微调权重来修正这一问题。

    参数：
        low_model_dir:  ft_model_dir 路径（如 .../low_noise_model/epoch_1），
                        提供时加载该目录全量权重到 low_noise_model。
        high_model_dir: ft_high_model_dir 路径，提供时同时转换 high_noise_model 并加载权重。
    """
    import gc
    from memory_module.model_with_memory import WanModelWithMemory

    logger.info("Converting low_noise_model to WanModelWithMemory (max_size=%d)...", memory_max_size)
    # 先在 CPU 建好新模型（skip_to_device=True），避免旧模型未释放时 GPU OOM
    _low_device = next(pipeline.low_noise_model.parameters()).device
    _low_dtype  = next(pipeline.low_noise_model.parameters()).dtype
    _new_low = WanModelWithMemory.from_wan_model(
        pipeline.low_noise_model,
        memory_layers=None,          # 全部 blocks（与 infer.py 默认行为一致）
        max_memory_size=memory_max_size,
        skip_to_device=True,
    )
    # 释放旧 GPU 模型后再搬迁新模型
    del pipeline.low_noise_model
    gc.collect()
    torch.cuda.empty_cache()
    pipeline.low_noise_model = _new_low.to(device=_low_device, dtype=_low_dtype)
    del _new_low
    gc.collect()
    torch.cuda.empty_cache()

    # 全参微调模式：从 ft checkpoint 重新加载全量权重（修复 .block. key 不匹配导致的权重丢失）
    if low_model_dir is not None:
        _low_state = _load_all_weights_from_hf_checkpoint(low_model_dir)
        if _low_state:
            _result = pipeline.low_noise_model.load_state_dict(_low_state, strict=False)
            logger.info(
                "Reloaded %d ft weights for low_noise_model from %s (missing=%d, unexpected=%d)",
                len(_low_state), low_model_dir,
                len(_result.missing_keys), len(_result.unexpected_keys),
            )
        else:
            logger.warning("No weights found in low_model_dir=%s", low_model_dir)

    # V5-B2-01 fix: LoRA 推理时 gate/nfp_head/latent_proj 已被单独保存，从此处加载
    # （full-param 模式下 low_model_dir 已覆盖，此分支仅供 LoRA 模式使用）
    if memory_ckpt_path is not None and os.path.exists(memory_ckpt_path):
        _mem_state = torch.load(memory_ckpt_path, map_location="cpu", weights_only=True)
        _result = pipeline.low_noise_model.load_state_dict(_mem_state, strict=False)
        logger.info(
            "V5-B2-01 fix: loaded %d memory weights from %s (missing=%d)",
            len(_mem_state), memory_ckpt_path, len(_result.missing_keys),
        )

    # dual 模型支持：若 high_model_dir 提供，同时转换 high_noise_model 并加载全量权重
    if high_model_dir is not None:
        logger.info(
            "Converting high_noise_model to WanModelWithMemory (dual mode, max_size=%d)...",
            memory_max_size,
        )
        _high_device = next(pipeline.high_noise_model.parameters()).device
        _high_dtype  = next(pipeline.high_noise_model.parameters()).dtype
        _new_high = WanModelWithMemory.from_wan_model(
            pipeline.high_noise_model,
            memory_layers=None,
            max_memory_size=memory_max_size,
            skip_to_device=True,
        )
        del pipeline.high_noise_model
        gc.collect()
        torch.cuda.empty_cache()
        pipeline.high_noise_model = _new_high.to(device=_high_device, dtype=_high_dtype)
        del _new_high
        gc.collect()
        torch.cuda.empty_cache()
        _high_state = _load_all_weights_from_hf_checkpoint(high_model_dir)
        if _high_state:
            _result = pipeline.high_noise_model.load_state_dict(_high_state, strict=False)
            logger.info(
                "Reloaded %d ft weights for high_noise_model from %s (missing=%d, unexpected=%d)",
                len(_high_state), high_model_dir,
                len(_result.missing_keys), len(_result.unexpected_keys),
            )
        else:
            logger.warning(
                "No weights found in high_model_dir=%s; "
                "high_noise_model starts from base weights.",
                high_model_dir,
            )

    logger.info("Pipeline conversion to WanModelWithMemory done.")
    return pipeline


def _patch_pipeline_memory(pipeline, memory_states, memory_value_states=None):
    """将 memory_states / memory_value_states 注入 pipeline 的模型 forward 中。

    SP 模式（model 有 _sp_memory_states 属性）：直接设置属性，
    _sp_dit_forward_with_memory 在 forward 时读取。
    非 SP 模式：monkey-patch model.forward（原有行为）。
    """
    if memory_states is None:
        return

    import functools
    from memory_module.model_with_memory import WanModelWithMemory

    def _apply(m, mem, mem_val):
        if not isinstance(m, WanModelWithMemory):
            return
        if hasattr(m, _SP_MEM_STATES_ATTR):
            # SP 模式：通过属性注入，不替换 forward
            setattr(m, _SP_MEM_STATES_ATTR, mem)
            setattr(m, _SP_MEM_VAL_ATTR,    mem_val)
        else:
            # 非 SP 模式：替换 forward（原有行为）
            def _make_patched(model, _mem, _mem_val):
                @functools.wraps(model.forward)
                def _patched(x, t, context, seq_len, y=None, dit_cond_dict=None):
                    _dev = next(model.parameters()).device
                    _mv  = _mem_val.to(_dev) if _mem_val is not None else None
                    return WanModelWithMemory.forward(
                        model, x, t, context, seq_len,
                        y=y, dit_cond_dict=dit_cond_dict,
                        memory_states=_mem.to(_dev),
                        memory_value_states=_mv,
                    )
                return _patched
            m._original_forward = m.forward
            m.forward = _make_patched(m, mem, mem_val)

    _apply(pipeline.low_noise_model,  memory_states, memory_value_states)
    _apply(pipeline.high_noise_model, memory_states, memory_value_states)


def _unpatch_pipeline_memory(pipeline):
    """还原 _patch_pipeline_memory 的注入（low_noise_model 和 high_noise_model）。"""
    def _restore(m):
        if hasattr(m, _SP_MEM_STATES_ATTR):
            # SP 模式：清空属性
            setattr(m, _SP_MEM_STATES_ATTR, None)
            setattr(m, _SP_MEM_VAL_ATTR,    None)
        elif hasattr(m, '_original_forward'):
            m.forward = m._original_forward
            del m._original_forward

    _restore(pipeline.low_noise_model)
    _restore(pipeline.high_noise_model)


def _update_memory_bank(bank, video, pipeline, device, clip_start_frame: int,
                        c2ws_plucker_emb=None, nfp_head=None, last_hidden_states=None,
                        chunk_id: int = 0):
    """用当前 clip 的 VAE latent 更新 MemoryBank（surprise-driven）。

    问题3修复：pose_emb 优先用 get_projected_frame_embs() 计算模型空间嵌入（dim=5120），
    与 MemoryCrossAttention 期待的 Key/Value 维度一致；退化时用 latent 均值（dim=16）。

    问题4修复（随问题3一起）：改用投影嵌入后，K/V 具有真正的视觉语义投影特征。

    问题5修复：surprise score 接口新增 nfp_head 参数，当前保留帧间 cosine distance 退化方案。
    # TODO：完整对齐需在推理时通过 forward hook 捕获 model.blocks[-1] 的 hidden_states，
    # 传入 nfp_head(hidden_states) 计算真实预测误差；当前用帧间 cosine distance 代理。

    Args:
        bank:               MemoryBank 实例
        video:              当前 clip 的视频帧（Tensor 或类 PIL 格式，与 VAE encode 兼容）
        pipeline:           WanI2V 管道实例（含 vae）
        device:             目标设备
        clip_start_frame:   当前 clip 在完整视频序列中的起始帧索引
        c2ws_plucker_emb:   可选，[1, C, lat_f, lat_h, lat_w]，来自 dit_cond_dict；
                            提供时用 get_projected_frame_embs() 计算 5120 维 pose_emb（问题3修复）
        nfp_head:           可选，NFPHead 实例；接口预留，当前退化为 cosine distance（问题5修复）
        last_hidden_states: 可选，[1, L, 5120]，forward hook 捕获的 model.blocks[-1] 输出；
                            提供时用 NFPHead 计算 clip-level surprise（M-1 修复）
    """
    import torch.nn.functional as F
    from memory_module.memory_bank import MemoryBank
    from memory_module.model_with_memory import WanModelWithMemory

    # FIX[B-02]：offload_model=True 时 VAE 可能已在 CPU，需先移回 device
    vae_device = next(pipeline.vae.model.parameters()).device
    if vae_device != device:
        pipeline.vae.model.to(device)

    with torch.no_grad():
        latent = pipeline.vae.encode([video.to(device)])[0]  # [z_dim, lat_f, h, w]

    lat_f = latent.shape[1]
    vae_stride_t = pipeline.vae_stride[0]

    # 问题3修复：计算 per-frame pose embedding（5120维，与 MemoryCrossAttention 对齐）
    model = pipeline.low_noise_model
    frame_embs = None
    if c2ws_plucker_emb is not None and isinstance(model, WanModelWithMemory):
        with torch.no_grad():
            # FIX: offload_model=True 时这些嵌入层可能已被卸载到 CPU；
            # 临时移到 device 执行计算（这几个层很小，移动开销可忽略）
            _pose_emb_layers = ['patch_embedding_wancamctrl',
                                'c2ws_hidden_states_layer1',
                                'c2ws_hidden_states_layer2']
            for _attr in _pose_emb_layers:
                if hasattr(model, _attr):
                    getattr(model, _attr).to(device)
            frame_embs = model.get_projected_frame_embs(
                c2ws_plucker_emb.to(device)
            )  # [lat_f, dim=5120]

    # BLOCK-2 修复：16维退化路径与 MemoryCrossAttention Linear(5120,5120) 不兼容，
    # 必须在此处拦截，避免将 dim=16 的 pose_emb 存入 bank 后在 retrieve() 时 crash。
    if frame_embs is None:
        logger.warning(
            "_update_memory_bank: c2ws_plucker_emb not provided or model is not "
            "WanModelWithMemory (model=%s); skipping bank update to prevent "
            "MemoryCrossAttention dim mismatch (expected dim=5120).",
            type(model).__name__,
        )
        return

    # M-1 修复：使用 NFPHead 计算 clip-level surprise（若 last_hidden_states 可用）
    from memory_module.nfp_head import NFPHead as _NFPHead
    clip_surprise = None
    if last_hidden_states is not None and hasattr(model, 'nfp_head') and model.nfp_head is not None:
        with torch.no_grad():
            # FIX: offload_model=True 时 nfp_head 可能已在 CPU
            model.nfp_head.to(device)
            _hs = last_hidden_states.to(device).to(
                next(model.nfp_head.parameters()).dtype
            )  # [1, L, 5120]
            _pred = model.nfp_head.forward(_hs)  # [1, z_dim=16]
            _actual = latent.float()[:, -1].mean(dim=[-2, -1]).unsqueeze(0)  # [1, 16] BLOCK-B 修复：最后帧空间均值，与 train_v2_stage1.py:770 一致
            clip_surprise = _NFPHead.compute_surprise(
                _pred.float(), _actual
            ).item()  # scalar
        logger.info("NFP clip surprise: %.4f", clip_surprise)

    # MODIFIED: F-03/F5 fix — 计算 per-frame visual embedding（5120维），
    # latent 空间均值投影到模型空间，作为 cross-attention V（视觉内容）
    visual_embs = None
    if isinstance(model, WanModelWithMemory) and hasattr(model, 'latent_proj'):
        with torch.no_grad():
            # FIX: offload_model=True 时 latent_proj 可能已在 CPU
            model.latent_proj.to(device)
            visual_embs = []
            for t_idx in range(lat_f):
                v_emb = model.get_projected_latent_emb(
                    latent[:, t_idx].to(device)
                )  # [dim=5120]
                visual_embs.append(v_emb.cpu())
        logger.debug("_update_memory_bank: computed %d visual_embs", lat_f)

    for t in range(lat_f):
        if clip_surprise is not None:
            # M-1 修复：使用 NFPHead clip-level surprise（per-clip 粒度对齐训练）
            surprise = clip_surprise
        elif t == 0:
            surprise = 1.0
        else:
            prev = latent[:, t - 1].flatten().float()
            curr = latent[:, t].flatten().float()
            cos_sim = F.cosine_similarity(prev.unsqueeze(0), curr.unsqueeze(0)).item()
            surprise = float(1.0 - cos_sim)

        pose_emb = frame_embs[t].cpu()  # [dim=5120]，由 BLOCK-2 保证必为 5120 维

        bank.update(
            pose_emb=pose_emb,
            latent=latent[:, t].cpu(),
            surprise_score=surprise,
            timestep=clip_start_frame + t * vae_stride_t,
            visual_emb=visual_embs[t] if visual_embs is not None else None,  # MODIFIED: F-03/F5 fix
            chunk_id=chunk_id,  # Feature 3 新增
        )

    logger.info("MemoryBank updated: %s", bank)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # ---- Step 1：处理 LoRA / 全参微调，准备最终 ckpt_dir ----
    if args.lora_path:
        args.ckpt_dir = _load_lora_and_prepare_ckpt(args)
    elif args.ft_model_dir:
        args.ckpt_dir = _load_ft_model_and_prepare_ckpt(args)

    # ---- Step 2：分布式初始化（与 inference_csgo.py 完全一致）----
    from wan.image2video import WanI2V
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video
    from wan.distributed.util import init_distributed_group
    from PIL import Image
    import torch.distributed as dist

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if rank == 0:
        logger.info("Rank 0 / World %d", world_size)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                rank=rank, world_size=world_size)
    if args.ulysses_size > 1:
        init_distributed_group()

    # ---- Step 3：加载 WanI2V 管道 ----
    cfg = WAN_CONFIGS["i2v-A14B"]

    # SP 模式下，当 use_memory=True，延迟到 _convert_pipeline_to_memory 之后手动应用；
    # 否则直接传给 WanI2V（与 inference_csgo.py 原有行为一致）。
    _use_sp   = args.ulysses_size > 1
    _use_fsdp = args.dit_fsdp
    if args.use_memory and _use_sp:
        # 避免 WanI2V._configure_model 对原始 WanModel 的 block 打 SP 补丁；
        # MemoryBlockWrapper 转换后再手动应用（见 Step 4）
        _wan_use_sp   = False
        _wan_dit_fsdp = False
    else:
        _wan_use_sp   = _use_sp
        _wan_dit_fsdp = _use_fsdp

    # Ulysses SP 要求 num_heads 能被 ulysses_size 整除
    if args.ulysses_size > 1:
        _num_heads = cfg.num_heads
        if _num_heads % args.ulysses_size != 0:
            _valid = sorted(i for i in range(1, _num_heads + 1) if _num_heads % i == 0)
            raise ValueError(
                f"Ulysses SP requires num_heads ({_num_heads}) % ulysses_size "
                f"({args.ulysses_size}) == 0. "
                f"Valid GPU counts: {_valid}"
            )

    wan_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=_wan_dit_fsdp,
        use_sp=_wan_use_sp,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---- Step 4：Memory 初始化（新增，仅当 --use_memory 时）----
    memory_bank = None
    if args.use_memory:
        from memory_module.memory_bank import MemoryBank

        logger.info(
            "Memory Bank enabled (max_size=%d). Converting pipeline to WanModelWithMemory...",
            args.memory_max_size,
        )
        _memory_ckpt_path = os.path.join(args.ckpt_dir, "memory_weights.pth")
        wan_i2v = _convert_pipeline_to_memory(
            wan_i2v,
            memory_max_size=args.memory_max_size,
            memory_ckpt_path=_memory_ckpt_path,    # V5-B2-01 fix: LoRA 模式下加载训练好的 memory 权重
            high_model_dir=args.ft_high_model_dir,  # dual 模型支持
            low_model_dir=args.ft_model_dir,        # 修复 .block. key 不匹配导致的全量权重丢失
        )

        # 在转换为 WanModelWithMemory 之后，手动应用 Ulysses SP
        if _use_sp:
            from memory_module.model_with_memory import WanModelWithMemory as _WMM
            logger.info("Applying Ulysses SP to WanModelWithMemory (use_sp=True, world_size=%d)", world_size)
            if isinstance(wan_i2v.low_noise_model, _WMM):
                wan_i2v.low_noise_model = _configure_memory_model_for_dist(
                    wan_i2v.low_noise_model, use_sp=True, device=device)
            if isinstance(wan_i2v.high_noise_model, _WMM):
                wan_i2v.high_noise_model = _configure_memory_model_for_dist(
                    wan_i2v.high_noise_model, use_sp=True, device=device)

        # T5-FSDP + SP 模式：SP init 会把 DiT 搬到 GPU，但 generate() 第一步就做 T5 FSDP
        # 文本编码，此时 DiT 和 T5 同时在 GPU 会 OOM（CUBLAS_STATUS_ALLOC_FAILED）。
        # 将 DiT 提前卸载到 CPU；generate(offload_model=True) 会在每个时间步按需搬回 GPU。
        if args.t5_fsdp and _use_sp:
            logger.info(
                "T5-FSDP+SP mode: offloading DiT to CPU before text encoding to avoid OOM"
            )
            for _attr in ('low_noise_model', 'high_noise_model'):
                _m = getattr(wan_i2v, _attr, None)
                if _m is not None and next(_m.parameters()).device.type == 'cuda':
                    _m.to('cpu')
            torch.cuda.empty_cache()

        memory_bank = MemoryBank(max_size=args.memory_max_size)
        logger.info("MemoryBank created: %s", memory_bank)

    # ---- Step 5：加载图像，生成视频 ----
    img = Image.open(args.image).convert("RGB")
    max_area = MAX_AREA_CONFIGS[args.size]

    if args.use_memory and memory_bank is not None:
        from memory_module.model_with_memory import WanModelWithMemory
        all_videos = []
        current_img = img

        # B4-1 修复：预加载 numpy 数组（循环外一次性 IO），plucker emb 改为每 clip 单独计算。
        # _c2ws_plucker_emb_for_bank 初始化为 None，循环内按 clip_idx 切片后覆盖。
        _poses_np = None
        _actions_np = None
        _intrinsics_np = None
        _h, _w = [int(x) for x in args.size.split("*")]
        _c2ws_plucker_emb_for_bank = None  # 保持变量名兼容后续代码，初始值 None
        # M-4 修复：广播 memory_states 初始化（多卡分布式推理时各 rank 保持一致）
        _broadcast_memory_states = None
        if args.action_path and os.path.isdir(args.action_path):
            try:
                import numpy as _np
                from pipeline.dataloader import build_dit_cond_dict as _build_dit_cond_dict
                _poses_np = _np.load(os.path.join(args.action_path, "poses.npy"))
                _actions_np = _np.load(os.path.join(args.action_path, "action.npy"))
                _intrinsics_np = _np.load(os.path.join(args.action_path, "intrinsics.npy"))
                logger.info(
                    "Loaded pose data: poses=%s actions=%s intrinsics=%s",
                    _poses_np.shape, _actions_np.shape, _intrinsics_np.shape,
                )
            except Exception as _e:
                logger.warning(
                    "Could not load pose data from action_path=%s: %s; "
                    "MemoryBank updates will be skipped.",
                    args.action_path, _e,
                )
        else:
            logger.warning(
                "action_path is None or not a directory (%s); "
                "MemoryBank updates will be skipped.", args.action_path,
            )

        for clip_idx in range(args.num_clips):
            logger.info("Generating clip %d/%d ...", clip_idx + 1, args.num_clips)

            # Feature 3：新 clip 开始前，已存储帧 age +1
            if clip_idx > 0:
                memory_bank.increment_age()

            # B4-1 修复：按 clip 计算当前帧段的 c2ws_plucker_emb（多 clip 时每 clip 用正确的 pose）
            _c2ws_plucker_emb_for_bank = None
            if _poses_np is not None:
                try:
                    clip_start_frame_idx = clip_idx * args.frame_num
                    clip_end_frame_idx = clip_start_frame_idx + args.frame_num
                    # 若 pose 数据足够长则取对应段，否则回退到可用的最后一段
                    if clip_end_frame_idx <= len(_poses_np):
                        _clip_poses = _poses_np[clip_start_frame_idx:clip_end_frame_idx]
                        _clip_actions = _actions_np[clip_start_frame_idx:clip_end_frame_idx]
                        _clip_intrinsics = _intrinsics_np[clip_start_frame_idx:clip_end_frame_idx]
                    else:
                        # pose 数据不足（单 clip 文件被复用）：回退到最后 frame_num 帧
                        _clip_poses = _poses_np[-args.frame_num:] if len(_poses_np) >= args.frame_num else _poses_np
                        _clip_actions = _actions_np[-args.frame_num:] if len(_actions_np) >= args.frame_num else _actions_np
                        _clip_intrinsics = _intrinsics_np[-args.frame_num:] if len(_intrinsics_np) >= args.frame_num else _intrinsics_np
                        logger.info(
                            "Clip %d: pose data shorter than expected (%d < %d), using last %d frames as fallback.",
                            clip_idx + 1, len(_poses_np), clip_end_frame_idx, len(_clip_poses),
                        )
                    _cond_clip = _build_dit_cond_dict(
                        poses=torch.from_numpy(_clip_poses).float(),
                        actions=torch.from_numpy(_clip_actions).float(),
                        intrinsics=torch.from_numpy(_clip_intrinsics).float(),
                        height=_h,
                        width=_w,
                    )
                    _c2ws_plucker_emb_for_bank = _cond_clip["c2ws_plucker_emb"][0]
                    logger.info(
                        "Clip %d: computed c2ws_plucker_emb for bank, shape=%s",
                        clip_idx + 1, tuple(_c2ws_plucker_emb_for_bank.shape),
                    )
                except Exception as _e:
                    logger.warning(
                        "Clip %d: failed to compute c2ws_plucker_emb: %s; memory bank updates will be skipped.",
                        clip_idx + 1, _e,
                    )

            # M-4：如果有广播来的 memory_states，优先使用（多卡一致性）
            memory_value_states_clip = None  # MODIFIED: F-03/F5 fix — 初始化 value states
            if _broadcast_memory_states is not None:
                memory_states, memory_value_states_clip = _broadcast_memory_states  # MODIFIED: F-03/F5 fix
                _broadcast_memory_states = None  # 消费后清空
                logger.info("Clip %d: using broadcast memory_states (M-4 fix)", clip_idx + 1)
            else:
                # 检索 memory（首 clip 时 bank 为空，memory_states=None）
                memory_states = None
                if memory_bank.size() > 0:
                    # HIGH-1 修复：用 get_projected_frame_embs 计算真实 pose query
                    model_lnm = wan_i2v.low_noise_model
                    if isinstance(model_lnm, WanModelWithMemory) and _c2ws_plucker_emb_for_bank is not None:
                        with torch.no_grad():
                            _qfe = model_lnm.get_projected_frame_embs(
                                _c2ws_plucker_emb_for_bank.to(device)
                            )  # [lat_f, dim=5120]
                        query_emb = _qfe[0].to(device)  # [5120]，当前 clip 第一帧 pose emb
                    else:
                        query_dim = memory_bank.frames[0].pose_emb.shape[0]
                        query_emb = torch.zeros(query_dim, device=device)
                        logger.warning("Clip %d: falling back to zero query (no pose data)", clip_idx + 1)
                    retrieved = memory_bank.retrieve(query_emb, top_k=4, device=device)
                    if retrieved is not None:
                        # MODIFIED: F-03/F5 fix — retrieve() 现在返回 (pose_embs, visual_embs) tuple
                        key_states, value_states = retrieved   # 各 [k, 5120]
                        memory_states = key_states.unsqueeze(0)               # [1, K, dim]
                        memory_value_states_clip = value_states.unsqueeze(0)  # [1, K, dim]
                        logger.info("Clip %d: retrieved %d memory frames.", clip_idx + 1, key_states.shape[0])
                    else:
                        memory_value_states_clip = None

            # M-1 修复：注册 forward hook 捕获 model.blocks[-1] hidden_states（供 NFPHead 使用）
            _nfp_captured_hs = {}
            _nfp_hook_handle = None
            _model_lnm = wan_i2v.low_noise_model
            if isinstance(_model_lnm, WanModelWithMemory):
                def _nfp_capture_hook(module, inp, out):
                    hs = out[0] if isinstance(out, (tuple, list)) else out
                    _nfp_captured_hs['hs'] = hs.detach().cpu()
                _nfp_hook_handle = _model_lnm.blocks[-1].register_forward_hook(_nfp_capture_hook)

            # MODIFIED: multi-clip action offset bugfix — 为每个 clip 按 clip_idx 切片
            # action/poses/intrinsics 写入临时目录，避免 generate() 总从 [0:frame_num] 读取
            _clip_action_start = clip_idx * args.frame_num
            _clip_action_end = _clip_action_start + args.frame_num
            _tmp_dir = None
            if _poses_np is not None and args.action_path:
                import tempfile as _tempfile_clip
                _tmp_dir = _tempfile_clip.mkdtemp(
                    prefix=f"lingbot_infer_clip{clip_idx}_r{rank}_"
                )
                # 按 clip_idx 切片，越界时 fallback 到最后 frame_num 帧
                if _clip_action_end <= len(_poses_np):
                    _tmp_poses = _poses_np[_clip_action_start:_clip_action_end]
                    _tmp_actions = _actions_np[_clip_action_start:_clip_action_end]
                    _tmp_intrinsics = _intrinsics_np[_clip_action_start:_clip_action_end]
                else:
                    _tmp_poses = _poses_np[-args.frame_num:]
                    _tmp_actions = _actions_np[-args.frame_num:]
                    _tmp_intrinsics = _intrinsics_np[-args.frame_num:]
                import numpy as _np_clip_tmp
                _np_clip_tmp.save(
                    os.path.join(_tmp_dir, "poses.npy"), _tmp_poses
                )
                _np_clip_tmp.save(
                    os.path.join(_tmp_dir, "action.npy"), _tmp_actions
                )
                _np_clip_tmp.save(
                    os.path.join(_tmp_dir, "intrinsics.npy"), _tmp_intrinsics
                )
                _action_path_for_clip = _tmp_dir
                logger.info(
                    "Clip %d: wrote per-clip action slice [%d:%d] to tmpdir %s (rank %d)",
                    clip_idx + 1, _clip_action_start, _clip_action_end, _tmp_dir, rank,
                )
            else:
                _action_path_for_clip = args.action_path

            # 注入 memory_states / memory_value_states 并生成
            # MODIFIED: F-03/F5 fix — 同时传入 visual_emb 作为 V
            _patch_pipeline_memory(wan_i2v, memory_states, memory_value_states_clip)
            try:
                video = wan_i2v.generate(
                    args.prompt,
                    current_img,
                    action_path=_action_path_for_clip,
                    max_area=max_area,
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver="unipc",
                    sampling_steps=args.sample_steps,
                    guide_scale=args.guide_scale,
                    seed=42 + clip_idx,
                    offload_model=True,
                )
            finally:
                _unpatch_pipeline_memory(wan_i2v)
                # M-1 修复：移除 forward hook
                if _nfp_hook_handle is not None:
                    _nfp_hook_handle.remove()
                    _nfp_hook_handle = None
                # 清理临时 action 目录
                if _tmp_dir is not None:
                    import shutil as _shutil_clip
                    _shutil_clip.rmtree(_tmp_dir, ignore_errors=True)
                    _tmp_dir = None
            _last_hs_for_nfp = _nfp_captured_hs.pop('hs', None)

            if rank == 0 and video is not None:
                # HIGH-3 修复：确保存入 all_videos 的是 torch.Tensor
                import numpy as _np_h3
                _video_tensor = torch.from_numpy(video.copy()) if isinstance(video, _np_h3.ndarray) else video
                all_videos.append(_video_tensor)
                # 更新 Memory Bank（BLOCK-1 修复：传入 c2ws_plucker_emb）
                _update_memory_bank(
                    bank=memory_bank,
                    video=video,
                    pipeline=wan_i2v,
                    device=device,
                    clip_start_frame=clip_idx * args.frame_num,
                    c2ws_plucker_emb=_c2ws_plucker_emb_for_bank,
                    last_hidden_states=_last_hs_for_nfp,
                    chunk_id=clip_idx,   # Feature 3 新增
                )
                # 使用最后一帧作为下一 clip 的初始帧
                # video shape: [C=3, T, H, W]，取最后时间步 [:, -1] → [C, H, W]
                from PIL import Image as PILImage
                import numpy as _np_frame
                last_frame_chw = video[:, -1]  # [C=3, H, W]
                if hasattr(last_frame_chw, 'cpu'):
                    last_frame_chw = last_frame_chw.cpu().float().numpy()
                # CHW → HWC，[-1,1] → [0,255]
                last_frame_hwc = last_frame_chw.transpose(1, 2, 0)
                last_frame_np = (last_frame_hwc * 127.5 + 127.5).clip(0, 255).astype(_np_frame.uint8)
                current_img = PILImage.fromarray(last_frame_np)
                logger.info("Clip %d: memory bank updated. Size=%d", clip_idx + 1, memory_bank.size())
                logger.info("Clip %d: bank stats: %s", clip_idx + 1, memory_bank.get_stats())

            # M-4 修复：广播 memory_states 给所有 rank（仅多卡 Ulysses 模式需要）
            # MODIFIED: F-03/F5 fix — 同时广播 memory_value_states（打包为 tuple 一起广播）
            if world_size > 1 and dist.is_initialized():
                # rank 0 预计算下一 clip 的 memory_states 和 memory_value_states 供广播
                if rank == 0 and memory_bank.size() > 0:
                    _m4_model = wan_i2v.low_noise_model
                    # N-03 修复：用下一 clip（clip_idx+1）的 pose 计算 M-4 广播的 memory query
                    _next_clip_idx = clip_idx + 1
                    _m4_plucker_emb = None
                    if _poses_np is not None and _next_clip_idx < args.num_clips:
                        try:
                            _nc_start = _next_clip_idx * args.frame_num
                            _nc_end = _nc_start + args.frame_num
                            if _nc_end <= len(_poses_np):
                                _nc_poses = _poses_np[_nc_start:_nc_end]
                                _nc_actions = _actions_np[_nc_start:_nc_end]
                                _nc_intrinsics = _intrinsics_np[_nc_start:_nc_end]
                            else:
                                _nc_poses = _poses_np[-args.frame_num:] if len(_poses_np) >= args.frame_num else _poses_np
                                _nc_actions = _actions_np[-args.frame_num:] if len(_actions_np) >= args.frame_num else _actions_np
                                _nc_intrinsics = _intrinsics_np[-args.frame_num:] if len(_intrinsics_np) >= args.frame_num else _intrinsics_np
                            _nc_cond = _build_dit_cond_dict(
                                poses=torch.from_numpy(_nc_poses).float(),
                                actions=torch.from_numpy(_nc_actions).float(),
                                intrinsics=torch.from_numpy(_nc_intrinsics).float(),
                                height=_h,
                                width=_w,
                            )
                            _m4_plucker_emb = _nc_cond["c2ws_plucker_emb"][0]
                        except Exception as _m4_e:
                            logger.warning("N-03: failed to compute next-clip plucker_emb for M-4: %s; falling back to current clip pose.", _m4_e)
                            _m4_plucker_emb = _c2ws_plucker_emb_for_bank
                    else:
                        # 已是最后一个 clip 或 pose 数据不足，退回当前 clip pose
                        _m4_plucker_emb = _c2ws_plucker_emb_for_bank
                    if isinstance(_m4_model, WanModelWithMemory) and _m4_plucker_emb is not None:
                        with torch.no_grad():
                            _m4_qfe = _m4_model.get_projected_frame_embs(
                                _m4_plucker_emb.to(device)
                            )
                        _m4_q = _m4_qfe[0].to(device)
                        _m4_retrieved = memory_bank.retrieve(_m4_q, top_k=4, device=device)
                        if _m4_retrieved is not None:
                            _m4_k, _m4_v = _m4_retrieved   # MODIFIED: F-03/F5 fix — unpack tuple
                            _m4_states = (
                                _m4_k.unsqueeze(0),  # [1, K, dim]
                                _m4_v.unsqueeze(0),  # [1, K, dim]
                            )
                        else:
                            _m4_states = None
                    else:
                        _m4_states = None
                else:
                    _m4_states = None
                # 广播 (memory_states, memory_value_states) tuple（broadcast_object_list 支持 None 和 Tensor）
                _m4_bcast = [_m4_states]
                dist.broadcast_object_list(_m4_bcast, src=0)
                _broadcast_memory_states = _m4_bcast[0]
                if _broadcast_memory_states is not None:
                    # 移到正确设备（tuple 中每项都需要 .to(device)）
                    _ms, _mv = _broadcast_memory_states
                    _broadcast_memory_states = (_ms.to(device), _mv.to(device))
            else:
                _broadcast_memory_states = None

        # 拼接所有 clips
        video = torch.cat(all_videos, dim=1) if all_videos else None  # BLOCK-A 修复：沿时间维度 T(dim=1) 拼接，[C, T*N, H, W]

    else:
        # 无 memory 路径：与 inference_csgo.py 完全一致
        video = wan_i2v.generate(
            args.prompt,
            img,
            action_path=args.action_path,
            max_area=max_area,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver="unipc",
            sampling_steps=args.sample_steps,
            guide_scale=args.guide_scale,
            seed=42,
            offload_model=False,
        )

    # ---- Step 6：保存输出（与 inference_csgo.py 完全一致）----
    if rank == 0 and video is not None:
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logger.info("Saved video → %s", args.save_file)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
