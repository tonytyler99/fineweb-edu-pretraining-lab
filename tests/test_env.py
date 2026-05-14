"""Environment sanity check.

Verifies the local PyTorch install can see a CUDA GPU and prints version /
device info so the user can confirm Phase 0 ended with a working CUDA wheel
on the RTX 3060 box. Intentionally kept assertion-light: we assert that CUDA
is available (hard requirement) but only *warn* if the GPU isn't a 3060,
since the same repo may later run on a different machine.
"""

from __future__ import annotations

import platform
import sys

import torch


def test_torch_imports_and_reports_versions() -> None:
    """Print interpreter, torch, and CUDA build versions for the test log."""
    print()
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    print(f"torch       : {torch.__version__}")
    print(f"torch.cuda  : {torch.version.cuda}")
    print(f"cuDNN       : {torch.backends.cudnn.version()}")


def test_cuda_is_available() -> None:
    """Hard requirement: the local install must expose a CUDA device."""
    assert torch.cuda.is_available(), (
        "CUDA is not available. If `uv sync` installed a CPU-only torch wheel, "
        "reinstall from the cu121 index:\n"
        "  uv pip install torch --index-url "
        "https://download.pytorch.org/whl/cu121 --reinstall"
    )


def test_gpu_is_visible_and_named() -> None:
    """Print the GPU name + total VRAM and softly check for the expected 2050."""
    device_count = torch.cuda.device_count()
    assert device_count >= 1, "No CUDA devices visible to torch."

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print()
    print(f"GPU count   : {device_count}")
    print(f"GPU[0] name : {name}")
    print(f"GPU[0] cap  : sm_{cap[0]}{cap[1]}")
    print(f"GPU[0] VRAM : {total_vram_gib:.2f} GiB")

    if "2050" not in name:
        print(
            f"WARN: expected an RTX 2050 on this box but found '{name}'. "
            "Update tests/test_env.py if this is intentional."
        )
