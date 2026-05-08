"""Benchmark harness for SD inference.

Runs a given configuration against the fixed prompt set, with proper warmup,
NVTX ranges around each pipeline stage, and per-image timing. Appends results
to a CSV that the analysis notebook reads.

Usage:
    python -m src.benchmark --config configs/runs/02_gpu_baseline_fp32.yaml

Optional flags for multi-GPU experiments:
    --gpu-tag        Suffix appended to run_id (e.g., 'gpu0')
    --prompt-subset  Comma-separated prompt IDs to override config's subset

This module assumes it is invoked from the project root so relative paths resolve.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch

from src.pipeline import build_pipeline
from src.utils import (
    capture_environment,
    load_yaml,
    set_all_seeds,
    write_env_snapshot,
)

# nvtx is optional - only needed for nsys profiling. Fall back to no-op if missing.
try:
    import nvtx
    _HAS_NVTX = True
except ImportError:
    _HAS_NVTX = False

    class _NVTXStub:
        @staticmethod
        def annotate(*args, **kwargs):
            class _Ctx:
                def __enter__(self_): return self_
                def __exit__(self_, *exc): return False
            return _Ctx()
    nvtx = _NVTXStub()


METRICS_CSV = Path("results/metrics.csv")
IMAGES_DIR = Path("results/images")

CSV_FIELDS = [
    "run_id", "prompt_id", "iteration", "wall_time_s", "device", "dtype",
    "attention_backend", "num_gpus", "batch_size", "num_inference_steps",
    "height", "width", "torch_version", "torch_cuda_version", "driver_version",
    "is_warmup",
]


def _select_prompts(prompts_cfg: dict, subset: Any) -> list[dict]:
    """Filter the master prompt list according to run config."""
    all_prompts = prompts_cfg["prompts"]
    if subset == "all" or subset is None:
        return all_prompts
    if isinstance(subset, list):
        return [p for p in all_prompts if p["id"] in subset]
    raise ValueError(f"Invalid prompts_subset: {subset!r}")


def _run_single(
    pipe,
    prompt_text: str,
    seed: int,
    inference_cfg: dict,
    device: str,
) -> tuple[float, Any]:
    """Run one inference call with NVTX ranges and return (wall_time, image)."""
    generator = torch.Generator(device=device if device != "cpu" else "cpu")
    generator.manual_seed(seed)

    # Synchronize before timing if on GPU to avoid measuring earlier work
    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with nvtx.annotate("full_inference", color="blue"):
        result = pipe(
            prompt=prompt_text,
            num_inference_steps=inference_cfg["num_inference_steps"],
            guidance_scale=inference_cfg["guidance_scale"],
            height=inference_cfg["height"],
            width=inference_cfg["width"],
            generator=generator,
        )

    if device != "cpu":
        torch.cuda.synchronize()

    wall = time.perf_counter() - t0
    return wall, result.images[0]


def _ensure_csv_header() -> None:
    """Create the metrics CSV with header row if it doesn't exist."""
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not METRICS_CSV.exists():
        with open(METRICS_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _append_row(row: dict[str, Any]) -> None:
    """Append a single row of metrics to the CSV."""
    with open(METRICS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def run_benchmark(
    run_config_path: str,
    prompts_config_path: str,
    gpu_tag: str | None = None,
    prompt_subset_override: str | None = None,
) -> None:
    """Main entry point: execute one full run from a config file.

    Optional args (used by the multi-GPU launcher):
        gpu_tag: appended to run_id so each process logs distinct rows
        prompt_subset_override: comma-separated prompt IDs to override config's subset
    """
    run_cfg = load_yaml(run_config_path)
    prompts_cfg = load_yaml(prompts_config_path)
    inference_cfg = prompts_cfg["inference"]

    # Apply gpu_tag suffix to run_id so multi-GPU rows are distinguishable
    if gpu_tag:
        run_cfg["run_id"] = f"{run_cfg['run_id']}_{gpu_tag}"

    # Override prompt subset from CLI if provided (used by multi-GPU launcher)
    if prompt_subset_override:
        run_cfg["prompts_subset"] = [
            p.strip() for p in prompt_subset_override.split(",")
        ]

    print(f"\n=== Run: {run_cfg['run_id']} ===")
    print(f"    {run_cfg['description']}")
    print(f"    device={run_cfg['device']}, dtype={run_cfg['dtype']}, "
          f"attention={run_cfg['attention_backend']}, ngpu={run_cfg['num_gpus']}")

    # Capture environment FIRST so even a build failure leaves a record
    snap = capture_environment()
    snapshot_dir = Path("results") / "env_snapshots" / run_cfg["run_id"]
    write_env_snapshot(snap, snapshot_dir)

    # Build pipeline
    set_all_seeds(0)
    print("    Building pipeline...")
    pipe = build_pipeline(run_cfg)

    # Filter prompts
    prompts = _select_prompts(prompts_cfg, run_cfg.get("prompts_subset", "all"))
    print(f"    Running {len(prompts)} prompts, "
          f"warmup={run_cfg['warmup_runs']}, timed={run_cfg['timed_runs']}")

    _ensure_csv_header()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    device = run_cfg["device"]

    # Warmup runs - first inference includes CUDA context init, kernel JIT,
    # and memory allocator warmup. Discard these timings.
    if run_cfg["warmup_runs"] > 0:
        print("    Warmup...")
        warm_prompt = prompts[0]
        for i in range(run_cfg["warmup_runs"]):
            wall, _ = _run_single(
                pipe, warm_prompt["text"], warm_prompt["seed"],
                inference_cfg, device,
            )
            print(f"      warmup {i+1}: {wall:.2f}s")

    # Timed runs
    print("    Timed runs...")
    for prompt in prompts:
        for i in range(run_cfg["timed_runs"]):
            wall, image = _run_single(
                pipe, prompt["text"], prompt["seed"],
                inference_cfg, device,
            )
            print(f"      {prompt['id']} iter {i+1}: {wall:.2f}s")

            # Save first iteration's image per prompt for quality eval
            if i == 0:
                img_path = IMAGES_DIR / f"{run_cfg['run_id']}_{prompt['id']}.png"
                image.save(img_path)

            _append_row({
                "run_id": run_cfg["run_id"],
                "prompt_id": prompt["id"],
                "iteration": i + 1,
                "wall_time_s": round(wall, 4),
                "device": run_cfg["device"],
                "dtype": run_cfg["dtype"],
                "attention_backend": run_cfg["attention_backend"],
                "num_gpus": run_cfg["num_gpus"],
                "batch_size": run_cfg["batch_size"],
                "num_inference_steps": inference_cfg["num_inference_steps"],
                "height": inference_cfg["height"],
                "width": inference_cfg["width"],
                "torch_version": snap.torch_version,
                "torch_cuda_version": snap.torch_cuda_version,
                "driver_version": snap.driver_version,
                "is_warmup": False,
            })

    print(f"    Done. Metrics appended to {METRICS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SD inference benchmark")
    parser.add_argument("--config", required=True, help="Path to run YAML")
    parser.add_argument(
        "--prompts", default="configs/prompts.yaml",
        help="Path to prompts YAML (default: configs/prompts.yaml)",
    )
    parser.add_argument(
        "--gpu-tag", default=None,
        help="Suffix appended to run_id for multi-GPU experiments (e.g., 'gpu0')",
    )
    parser.add_argument(
        "--prompt-subset", default=None,
        help="Comma-separated prompt IDs to override config's prompts_subset (e.g., 'p01,p03,p05')",
    )
    args = parser.parse_args()
    run_benchmark(
        args.config,
        args.prompts,
        gpu_tag=args.gpu_tag,
        prompt_subset_override=args.prompt_subset,
    )


if __name__ == "__main__":
    main()
