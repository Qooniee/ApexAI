#!/usr/bin/env python3
"""PyTorch/CUDA environment check script.

Verifies that PyTorch and CUDA are properly set up in the Docker environment.
"""

import platform
import subprocess
import sys


def check_python():
    """Check Python environment information."""
    print("=" * 50)
    print("Python Environment")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()


def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("=" * 50)
    print("PyTorch Environment")
    print("=" * 50)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch compiled with CUDA: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                memory_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  Memory: {memory_gb:.1f} GB")

            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        else:
            print("CUDA is not available")
    except ImportError:
        print("PyTorch is not installed")
    print()


def check_torchvision():
    """Check TorchVision installation."""
    print("=" * 50)
    print("TorchVision")
    print("=" * 50)

    try:
        import torchvision

        print(f"TorchVision version: {torchvision.__version__}")
    except ImportError:
        print("TorchVision is not installed")
    print()


def check_cuda_toolkit():
    """Check CUDA toolkit version and configuration."""
    print("=" * 50)
    print("CUDA Toolkit")
    print("=" * 50)

    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("NVCC (CUDA Compiler):")
            print(result.stdout)
        else:
            print("NVCC not found")
    except FileNotFoundError:
        print("NVCC not found")

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("NVIDIA-SMI Output:")
            print(result.stdout)
        else:
            print("nvidia-smi not found")
    except FileNotFoundError:
        print("nvidia-smi not found")
    print()


def run_simple_test():
    """Run simple PyTorch operations to verify functionality."""
    print("=" * 50)
    print("Simple PyTorch Test")
    print("=" * 50)

    try:
        import torch

        # CPU test
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        torch.mm(x, y)
        print("CPU tensor operation: OK")

        # GPU test if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print("GPU tensor operation: OK")
            print(f"Result tensor shape: {z_gpu.shape}")
            print(f"Result tensor device: {z_gpu.device}")
        else:
            print("GPU test skipped (CUDA not available)")

    except Exception as e:
        print(f"Test failed: {e}")
    print()


def main():
    """Run all environment checks."""
    print("PyTorch/CUDA Environment Check")
    print("=" * 50)

    check_python()
    check_pytorch()
    check_torchvision()
    check_cuda_toolkit()
    run_simple_test()

    print("Environment check complete!")


if __name__ == "__main__":
    main()
