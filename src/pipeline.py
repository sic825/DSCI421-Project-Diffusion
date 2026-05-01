"""Stable Diffusion pipeline factory.

Centralizes the logic for constructing an SD pipeline from a config dict so every
benchmark goes through the same code path. Keeps configuration logic out of
benchmark.py.
"""
from __future__ import annotations

from typing import Any

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# SD 1.5 weights from RunwayML are no longer hosted on HF as of 2024;
# use the community mirror. If this fails on the HPC, swap in a local path.
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.lower().replace("-", "").replace("_", "")
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {name}. Expected one of {list(_DTYPE_MAP)}.")
    return _DTYPE_MAP[key]


def _set_attention_backend(pipe: StableDiffusionPipeline, backend: str) -> None:
    """Configure attention backend on the UNet.

    Backends:
        vanilla : default PyTorch attention (no optimization)
        sdpa    : torch.nn.functional.scaled_dot_product_attention (built-in)
        xformers: xformers memory-efficient attention (requires xformers package)
    """
    backend = backend.lower()
    unet = pipe.unet

    if backend == "vanilla":
        # Force the legacy attention implementation. Diffusers calls this
        # AttnProcessor (without "2_0"), which routes through manual matmul ops.
        from diffusers.models.attention_processor import AttnProcessor
        unet.set_attn_processor(AttnProcessor())

    elif backend == "sdpa":
        # AttnProcessor2_0 routes through F.scaled_dot_product_attention.
        from diffusers.models.attention_processor import AttnProcessor2_0
        unet.set_attn_processor(AttnProcessor2_0())

    elif backend == "xformers":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except (ModuleNotFoundError, ImportError) as e:
            raise RuntimeError(
                "xformers backend requested but xformers is not installed. "
                "Install with: pip install xformers==0.0.27.post2"
            ) from e

    else:
        raise ValueError(
            f"Unknown attention_backend: {backend}. "
            "Expected one of: vanilla, sdpa, xformers."
        )


def build_pipeline(config: dict[str, Any]) -> StableDiffusionPipeline:
    """Construct an SD pipeline from a run config dict.

    Required keys: device, dtype, attention_backend
    Optional keys: model_id (defaults to SD 1.5)
    """
    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    dtype = _resolve_dtype(config["dtype"])
    device = config["device"]

    # CPU runs must use FP32 - FP16 on CPU is unsupported in most ops.
    if device == "cpu" and dtype != torch.float32:
        raise ValueError(
            f"CPU device requires FP32, got {config['dtype']}. "
            "Configure CPU runs with dtype=float32."
        )

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # we control the prompts; skip the latency cost
        requires_safety_checker=False,
    )

    # Standardize on DPM-Solver++ for deterministic step count comparison.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    _set_attention_backend(pipe, config["attention_backend"])

    # Disable progress bars so they don't spam logs during timed runs
    pipe.set_progress_bar_config(disable=True)

    return pipe
