# 421-project-diffusion

Profiling GPU acceleration techniques for Stable Diffusion v1.5 inference on the Lehigh ECE HPC cluster. Final project for DSCI 421: Accelerating Computing for Deep Learning, Spring 2026.

**Team:** Simon Chen (sic825), Vincent Caruso (vdc225), Matt Olson (mjo225)

## Project structure

```
configs/        YAML config files defining prompts, seeds, and run matrix
src/            Python source for the pipeline, benchmark harness, and quality eval
scripts/        Shell scripts that wrap benchmark.py for SLURM-style HPC runs
notebooks/      Jupyter notebooks for environment checks and results analysis
results/        Output directory for nsys reports, generated images, metrics CSV
report/         Placeholder for the LaTeX report (kept in Overleaf)
```

## Hardware target

Lehigh ECE HPC node (`hpc05.ece.lehigh.edu`):

- 2x NVIDIA GeForce RTX 2080 Ti (11 GB VRAM each, NVLink ~25.78 GB/s per link, 2 links per GPU)
- Driver 565.57.01, CUDA runtime 12.7, nvcc 12.4
- Anaconda Python 3.12.2 in `/opt/anaconda/Anaconda3-2024.10-1`
- NVIDIA HPC SDK 24.11 at `/opt/nvidia/hpc_sdk/Linux_x86_64/24.11`

## Environment setup

```bash
# Create a dedicated conda environment so we don't touch base
conda env create -f environment.yml
conda activate sd-profiling

# Verify CUDA is visible to PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
```

Expected output: `True 12.4 2`

## Methodology

We compare three categories of acceleration applied incrementally to the same Stable Diffusion 1.5 inference workload, holding model weights, prompts, seeds, sampler, and resolution constant across all runs.

1. **Precision sweep:** FP32 (baseline) -> FP16 -> BF16. INT8 dropped from the proposal pending feasibility check.
2. **Attention backend:** vanilla -> PyTorch SDPA memory-efficient -> xformers, on top of the best precision setting.
3. **Multi-GPU throughput:** single-GPU vs dual-GPU batched inference using `accelerate`, measuring images-per-second rather than single-image latency.

Profiling uses Nsight Systems for end-to-end timeline and per-kernel timing with NVTX ranges around each pipeline stage (text encode, denoising loop, VAE decode). One dominant attention kernel is profiled with Nsight Compute for occupancy and memory throughput.

Quality is measured with CLIP score across a fixed prompt set with fixed seeds, plus visual side-by-side grids in the report.

## Running benchmarks

```bash
# Single config run
python -m src.benchmark --config configs/runs/01_cpu_baseline.yaml

# Sweep all configs
bash scripts/run_all.sh

# Profile with Nsight Systems
bash scripts/profile_nsys.sh configs/runs/04_fp16_sdpa.yaml
```

All runs append to `results/metrics.csv`. The analysis notebook reads this CSV plus the `.nsys-rep` files for plots.

## Reproducibility

Every run captures the full environment (driver version, CUDA version, PyTorch version, GPU model, NVLink status, environment variables) into the metrics CSV alongside timing data. Seeds are set deterministically. Any deviation from the reference output is logged.

## Status

In progress. Deadline: Friday, May 8, 2026 (report + code), Monday, May 11, 2026 (demo video).
