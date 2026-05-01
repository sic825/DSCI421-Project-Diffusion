"""Utilities: seeding, environment capture, config loading.

Keep this module dependency-light so it can be imported on the CPU baseline run
without dragging in CUDA-specific imports unnecessarily.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class EnvSnapshot:
    """Captured at the start of every benchmark run for reproducibility."""
    hostname: str
    python_version: str
    platform: str
    torch_version: str = ""
    torch_cuda_version: str = ""
    cuda_runtime_version: str = ""
    nvcc_version: str = ""
    driver_version: str = ""
    gpu_names: list[str] = field(default_factory=list)
    gpu_count: int = 0
    nvlink_active: bool = False
    relevant_env_vars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_all_seeds(seed: int) -> None:
    """Set all RNG seeds. Note: full determinism for diffusion models is hard
    because of cuDNN nondeterminism. We accept some run-to-run variance and
    rely on multiple timed runs averaged together."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _shell(cmd: list[str]) -> str:
    """Run a shell command and return stdout, swallowing errors."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def capture_environment() -> EnvSnapshot:
    """Snapshot the environment for reproducibility logging."""
    snap = EnvSnapshot(
        hostname=platform.node(),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )

    try:
        import torch
        snap.torch_version = torch.__version__
        snap.torch_cuda_version = torch.version.cuda or ""
        snap.gpu_count = torch.cuda.device_count()
        snap.gpu_names = [torch.cuda.get_device_name(i) for i in range(snap.gpu_count)]
    except ImportError:
        pass

    nvcc = _shell(["nvcc", "--version"])
    if nvcc:
        # Parse the line "Cuda compilation tools, release 12.4, V12.4.131"
        for line in nvcc.splitlines():
            if "release" in line:
                snap.nvcc_version = line.strip()
                break

    smi = _shell(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        snap.driver_version = smi.splitlines()[0].strip()

    nvlink = _shell(["nvidia-smi", "nvlink", "--status"])
    snap.nvlink_active = "GB/s" in nvlink

    # Capture env vars that might affect performance
    relevant_vars = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    snap.relevant_env_vars = {
        k: os.environ.get(k, "") for k in relevant_vars if os.environ.get(k)
    }

    return snap


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def write_env_snapshot(snap: EnvSnapshot, output_dir: Path) -> Path:
    """Write the environment snapshot to JSON for later analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "env_snapshot.json"
    with open(out, "w") as f:
        json.dump(snap.to_dict(), f, indent=2)
    return out
