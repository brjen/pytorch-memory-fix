#!/usr/bin/env python3
"""
Benchmark: PyTorch Memory Creep with and without MALLOC_MMAP fix.

Demonstrates that glibc arena fragmentation causes RSS to grow permanently
when loading/unloading large PyTorch models, and that setting
MALLOC_MMAP_THRESHOLD_=65536 completely eliminates the issue.

Usage:
    python benchmark.py              # Run with fix applied
    python benchmark.py --no-fix     # Run WITHOUT fix (watch RSS grow)
    python benchmark.py --cycles 20  # Custom cycle count
    python benchmark.py --model bert # Use a specific model

Requirements:
    pip install torch transformers psutil
"""

import argparse
import gc
import os
import sys
import time
import ctypes

# Apply fix BEFORE any PyTorch imports (if not --no-fix)
parser = argparse.ArgumentParser(description="PyTorch memory creep benchmark")
parser.add_argument("--no-fix", action="store_true", help="Run WITHOUT the mmap fix")
parser.add_argument("--cycles", type=int, default=10, help="Number of load/unload cycles (default: 10)")
parser.add_argument("--model", default="auto", help="Model to test: auto, bert, gpt2, resnet (default: auto)")
args = parser.parse_args()

if not args.no_fix:
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '65536'
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '65536'

import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def get_rss_mb():
    """Get current process RSS in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    else:
        # Fallback for Linux
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except (FileNotFoundError, PermissionError):
            return 0.0


def malloc_trim():
    """Call glibc malloc_trim to return free heap pages to OS."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Not on Linux/glibc


def load_model(model_name):
    """Load a PyTorch model into CPU RAM."""
    if model_name == "bert" and HAS_TRANSFORMERS:
        return AutoModel.from_pretrained("bert-base-uncased")
    elif model_name == "gpt2" and HAS_TRANSFORMERS:
        return AutoModel.from_pretrained("gpt2")
    elif model_name == "resnet":
        from torchvision import models
        return models.resnet152(pretrained=False)
    else:
        # Fallback: create a large random tensor model
        # This simulates a ~500MB model in RAM
        layers = []
        for _ in range(10):
            layers.append(torch.nn.Linear(4096, 4096))
        return torch.nn.Sequential(*layers)


def unload_model(model):
    """Unload model and reclaim memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    malloc_trim()


def main():
    mode = "WITH FIX" if not args.no_fix else "WITHOUT FIX"
    print(f"\n{'='*60}")
    print(f"PyTorch Memory Creep Benchmark — {mode}")
    print(f"{'='*60}")
    print(f"Cycles: {args.cycles}")
    print(f"MALLOC_MMAP_THRESHOLD_: {os.environ.get('MALLOC_MMAP_THRESHOLD_', 'not set')}")
    print(f"MALLOC_TRIM_THRESHOLD_: {os.environ.get('MALLOC_TRIM_THRESHOLD_', 'not set')}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Detect model
    model_name = args.model
    if model_name == "auto":
        if HAS_TRANSFORMERS:
            model_name = "bert"
        else:
            model_name = "synthetic"
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")

    baseline_rss = get_rss_mb()
    print(f"{'Cycle':<8} {'Load RSS (MB)':<16} {'Unload RSS (MB)':<18} {'Drift (MB)':<12}")
    print(f"{'-'*54}")

    for i in range(args.cycles):
        # Load
        model = load_model(model_name)
        time.sleep(0.5)  # Let allocations settle
        load_rss = get_rss_mb()

        # Unload
        unload_model(model)
        time.sleep(1.0)  # Give OS time to reclaim pages
        unload_rss = get_rss_mb()
        drift = unload_rss - baseline_rss

        print(f"{i+1:<8} {load_rss:<16.1f} {unload_rss:<18.1f} {drift:+.1f}")

    final_rss = get_rss_mb()
    total_drift = final_rss - baseline_rss

    print(f"\n{'='*60}")
    print(f"Baseline RSS:    {baseline_rss:.1f} MB")
    print(f"Final RSS:       {final_rss:.1f} MB")
    print(f"Total drift:     {total_drift:+.1f} MB")
    print(f"Avg drift/cycle: {total_drift/args.cycles:+.1f} MB")

    if total_drift > 50:
        print(f"\n⚠️  RSS drifted {total_drift:.0f}MB over {args.cycles} cycles.")
        print(f"   Run again with the fix: python benchmark.py")
    else:
        print(f"\n✅ RSS stable — drift is within noise ({total_drift:.1f}MB).")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
