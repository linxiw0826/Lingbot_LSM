"""
wandb_utils.py -- W&B Observability for Lingbot Memory Training

统一封装 W&B 初始化、指标记录、梯度健康检查、crash recovery。
避免在训练脚本中分散 wandb 调用代码。

用法:
    from pipeline.common.wandb_utils import WandBLogger
    wb = WandBLogger(args, accelerator)
    wb.log_step(step, loss_dict, model)
    wb.finish()

SLURM 集成:
  - WANDB_DIR/WANDB_CACHE_DIR/WANDB_CONFIG_DIR/WANDB_DATA_DIR 全部指向项目目录
  - 自动检测 .netrc 中的 WANDB_API_KEY 作为 fallback
  - log_crash() 接收 SLURM 日志路径并上传到 run files

参考:
    - DPO 项目的 wandb 集成经验（conversation b93690fc）
"""

import logging
import netrc
import os
import time
import traceback
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WandBLogger:
    """W&B logging wrapper for Lingbot Memory training.

    Handles:
    - Early init with proper env vars for SLURM
    - Per-step loss/metric logging (called BEFORE optimizer.zero_grad)
    - Gradient norm tracking for memory branch
    - Gate value tracking
    - Numerical health monitoring (NaN/Inf detection)
    - Crash recovery log upload (SLURM log path)
    """

    def __init__(
        self,
        args,
        accelerator=None,
        project: str = "lingbot-memory",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        mode: str = "online",
        log_every_steps: int = 10,
    ):
        self.enabled = True
        self.log_every_steps = log_every_steps
        self._run = None
        self._start_time = time.time()

        # Only init on main process
        is_main = (accelerator is None) or accelerator.is_main_process
        if not is_main:
            self.enabled = False
            return

        # Override from args if available
        project = getattr(args, 'wandb_project', project)
        entity = getattr(args, 'wandb_entity', entity)
        run_name = getattr(args, 'wandb_run_name', run_name)
        mode = getattr(args, 'wandb_mode', mode)
        log_every_steps = getattr(args, 'log_every_steps', log_every_steps)
        self.log_every_steps = log_every_steps

        if mode == "disabled":
            self.enabled = False
            logger.info("W&B disabled by --wandb_mode=disabled")
            return

        try:
            import wandb

            # SLURM env: redirect ALL W&B directories to project dir
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
            wandb_dir = os.path.join(project_root, "wandb_runs")
            wandb_cache_dir = os.path.join(wandb_dir, ".cache")
            wandb_config_dir = os.path.join(wandb_dir, ".config")
            wandb_data_dir = os.path.join(wandb_dir, ".data")
            for path in [wandb_dir, wandb_cache_dir, wandb_config_dir, wandb_data_dir]:
                os.makedirs(path, exist_ok=True)
            os.environ.setdefault("WANDB_DIR", wandb_dir)
            os.environ.setdefault("WANDB_CACHE_DIR", wandb_cache_dir)
            os.environ.setdefault("WANDB_CONFIG_DIR", wandb_config_dir)
            os.environ.setdefault("WANDB_DATA_DIR", wandb_data_dir)

            # API key fallback: check .netrc if WANDB_API_KEY not set
            if "WANDB_API_KEY" not in os.environ:
                netrc_key = self._extract_wandb_api_key_from_netrc()
                if netrc_key:
                    os.environ["WANDB_API_KEY"] = netrc_key
                    logger.info("Exported WANDB_API_KEY from ~/.netrc for W&B auth fallback.")

            config = {k: v for k, v in vars(args).items() if not k.startswith('_')}
            for env_key in ["SLURM_JOB_ID", "SLURM_NODELIST", "CUDA_VISIBLE_DEVICES"]:
                env_val = os.environ.get(env_key)
                if env_val:
                    config[f"env_{env_key.lower()}"] = env_val

            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                mode=mode,
                config=config,
                reinit=True,
            )
            logger.info("W&B initialized: project=%s, run=%s, mode=%s",
                        project, self._run.name, mode)

        except Exception as e:
            logger.warning("W&B init failed (non-fatal): %s", e)
            self.enabled = False

    def log_step(
        self,
        step: int,
        loss_dict: Dict[str, float],
        model: Optional[nn.Module] = None,
        lr: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log metrics at a training step.

        IMPORTANT: This must be called AFTER backward() but BEFORE
        optimizer.zero_grad(), so that gradient norms are still available.

        Args:
            step: global step
            loss_dict: {"loss/total": ..., "loss/diffusion": ..., etc.}
            model: model for gradient norm computation (optional)
            lr: current learning rate (optional)
            extra: additional metrics to log
        """
        if not self.enabled or step % self.log_every_steps != 0:
            return

        metrics = dict(loss_dict)

        # Learning rate
        if lr is not None:
            metrics["train/lr"] = lr

        # Elapsed time
        metrics["train/elapsed_seconds"] = time.time() - self._start_time
        metrics["train/step"] = step

        # Gradient norms + gate values for memory branch
        if model is not None:
            grad_norms = self._compute_grad_norms(model)
            metrics.update(grad_norms)
            metrics.update(self._collect_memory_diagnostics(model))

        # Numerical health
        health = self._check_numerical_health(loss_dict)
        metrics.update(health)

        # Extra
        if extra:
            metrics.update(extra)

        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning("W&B log failed at step %d: %s", step, e)

    def log_memory_stats(self, bank_stats: Dict[str, float], step: int) -> None:
        """Log MemoryBank statistics."""
        if not self.enabled or step % self.log_every_steps != 0:
            return
        try:
            import wandb
            wandb.log(bank_stats, step=step)
        except Exception as e:
            logger.warning("W&B memory stats log failed: %s", e)

    def log_crash(self, exc: Exception, log_path: Optional[str] = None) -> None:
        """Log crash info to W&B for post-mortem analysis.

        Args:
            exc: the exception that caused the crash
            log_path: path to SLURM log file to upload (e.g. slurm-12345.out)
        """
        if not self.enabled:
            return
        try:
            import wandb
            crash_info = {
                "crash/type": type(exc).__name__,
                "crash/message": str(exc)[:500],
                "crash/traceback": traceback.format_exc()[:2000],
            }
            wandb.log(crash_info)

            if log_path and os.path.exists(log_path):
                wandb.save(log_path, policy="now")
                logger.info("Uploaded crash log to W&B: %s", log_path)
            else:
                # Try auto-detect SLURM log
                slurm_job_id = os.environ.get('SLURM_JOB_ID')
                if slurm_job_id:
                    for candidate_dir in ['.', os.environ.get('SLURM_SUBMIT_DIR', '.')]:
                        candidate = os.path.join(candidate_dir, f"slurm-{slurm_job_id}.out")
                        if os.path.exists(candidate):
                            wandb.save(candidate, policy="now")
                            logger.info("Auto-uploaded SLURM log to W&B: %s", candidate)
                            break

        except Exception as e:
            logger.warning("W&B crash log failed: %s", e)

    def finish(self) -> None:
        """Finish the W&B run."""
        if not self.enabled or self._run is None:
            return
        try:
            import wandb
            wandb.finish()
            logger.info("W&B run finished.")
        except Exception as e:
            logger.warning("W&B finish failed: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_wandb_api_key_from_netrc() -> Optional[str]:
        """Parse ~/.netrc and export the W&B API key when available."""
        netrc_path = os.path.expanduser("~/.netrc")
        if not os.path.exists(netrc_path):
            return None

        try:
            auth_data = netrc.netrc(netrc_path)
        except Exception as exc:
            logger.warning("Failed to parse ~/.netrc for W&B auth fallback: %s", exc)
            return None

        for machine in ("api.wandb.ai", "wandb.ai"):
            creds = auth_data.authenticators(machine)
            if not creds:
                continue
            password = creds[2]
            if password:
                return password

        return None

    @staticmethod
    def _compute_grad_norms(model: nn.Module) -> Dict[str, float]:
        """Compute gradient norms for memory-related parameters.

        MUST be called AFTER backward() and BEFORE optimizer.zero_grad().
        """
        norms = {}
        memory_grad_norm_sq = 0.0
        memory_param_count = 0
        nfp_grad_norm_sq = 0.0
        nfp_param_count = 0
        gate_values = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Track gate values (always, regardless of grad)
            if name.endswith("gate"):
                gate_values[f"gate/{name}"] = param.item()

            if param.grad is None:
                continue

            if "memory_cross_attn" in name or "memory_norm" in name:
                memory_grad_norm_sq += param.grad.float().norm().item() ** 2
                memory_param_count += 1
            elif "nfp_head" in name:
                nfp_grad_norm_sq += param.grad.float().norm().item() ** 2
                nfp_param_count += 1

        norms["grad/memory_cross_attn_norm"] = memory_grad_norm_sq ** 0.5
        norms["grad/nfp_head_norm"] = nfp_grad_norm_sq ** 0.5
        norms["grad/memory_param_count"] = float(memory_param_count)
        norms["grad/nfp_param_count"] = float(nfp_param_count)
        norms.update(gate_values)

        return norms

    @staticmethod
    def _collect_memory_diagnostics(model: nn.Module) -> Dict[str, float]:
        """Collect runtime diagnostics emitted by MemoryCrossAttention modules."""
        attn_norms = []
        gate_values = []

        for module in model.modules():
            attn_norm = getattr(module, "_last_attn_out_norm", None)
            gate_val = getattr(module, "_last_gate_value", None)
            if attn_norm is None or gate_val is None:
                continue
            attn_norms.append(float(attn_norm))
            gate_values.append(float(gate_val))

        if not attn_norms:
            return {}

        return {
            "memory/attn_out_norm_mean": sum(attn_norms) / len(attn_norms),
            "memory/attn_out_norm_max": max(attn_norms),
            "memory/runtime_gate_mean": sum(gate_values) / len(gate_values),
            "memory/runtime_gate_max": max(gate_values),
            "memory/runtime_block_count": float(len(attn_norms)),
        }

    @staticmethod
    def _check_numerical_health(loss_dict: Dict[str, float]) -> Dict[str, float]:
        """Check for NaN/Inf in losses."""
        import math
        health = {"health/has_nan": 0.0, "health/has_inf": 0.0}
        for key, val in loss_dict.items():
            if isinstance(val, (int, float)):
                if math.isnan(val):
                    health["health/has_nan"] = 1.0
                    logger.warning("NaN detected in %s", key)
                if math.isinf(val):
                    health["health/has_inf"] = 1.0
                    logger.warning("Inf detected in %s", key)
        return health
