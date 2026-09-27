"""Report the Python / PyTorch / GPU environment.

Run from the project root:  python scripts/check_env.py

Does not install anything. If PyTorch is missing it says so and exits normally.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"Python:   {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    try:
        import torch
    except ImportError:
        print("PyTorch:  not installed")
        return 0

    print(f"PyTorch:  {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    print(f"torch.version.hip:  {hip}")
    print(f"torch.version.cuda: {cuda}")

    try:
        available = torch.cuda.is_available()
    except Exception as exc:  # broken driver installs can raise here
        print(f"torch.cuda.is_available() raised: {exc!r}")
        return 0

    print(f"torch.cuda.is_available(): {available}")
    if available:
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        backend = "ROCm" if hip else "CUDA" if cuda else "CPU-only build"
        print(f"No GPU visible to PyTorch ({backend}); CPU fallback will be used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
