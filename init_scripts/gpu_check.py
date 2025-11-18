#!/usr/bin/env python3
"""AxilDrive GPU Quick Check.

Quick GPU functionality verification for setup validation (simplified version).
"""

import os
import sys
import time

import torch


def check_gpu_basic():
    """Basic GPU check."""
    print("🔍 GPU Basic Check...")

    results = {}

    # 1. Check CUDA availability
    cuda_available = torch.cuda.is_available()
    results["cuda_available"] = cuda_available
    status = "✅" if cuda_available else "❌"
    print(f"  {status} CUDA Available: {cuda_available}")

    if not cuda_available:
        print("  ⚠️  CPU mode - GPU acceleration disabled")
        return results

    # 2. GPU count and device information
    device_count = torch.cuda.device_count()
    results["device_count"] = device_count
    print(f"  ✅ GPU Count: {device_count}")

    if device_count > 0:
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        results["gpu_name"] = gpu_name
        results["gpu_memory_gb"] = memory_gb
        print(f"  ✅ GPU 0: {gpu_name}")
        print(f"  ✅ GPU Memory: {memory_gb:.1f}GB")

        # 3. Simple tensor operation test
        try:
            device = torch.device("cuda:0")
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor)
            results["tensor_ops"] = result.is_cuda
            print("  ✅ Tensor Operations: Success")
        except Exception as e:
            results["tensor_ops"] = False
            print(f"  ❌ Tensor Operations: Failed ({e})")

        # 4. Simple neural network operation test
        try:
            model = torch.nn.Linear(100, 10).to(device)
            input_data = torch.randn(32, 100).to(device)
            output = model(input_data)
            results["neural_ops"] = output.is_cuda
            print("  ✅ Neural Network Ops: Success")
        except Exception as e:
            results["neural_ops"] = False
            print(f"  ❌ Neural Network Ops: Failed ({e})")

    return results


def check_gpu_environment():
    """Check GPU environment variables."""
    print("\n🌍 GPU Environment Check...")

    env_vars = [
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_VISIBLE_DEVICES",
    ]

    for var in env_vars:
        value = os.getenv(var, "Not Set")
        print(f"  📋 {var}: {value}")


def quick_performance_test():
    """Quick performance test."""
    if not torch.cuda.is_available():
        return False

    print("\n⚡ Quick Performance Test...")

    try:
        device = torch.device("cuda:0")
        size = 1000

        # CPU computation
        cpu_a = torch.randn(size, size)
        cpu_b = torch.randn(size, size)

        start_time = time.time()
        torch.matmul(cpu_a, cpu_b)
        cpu_time = time.time() - start_time

        # GPU computation
        gpu_a = cpu_a.to(device)
        gpu_b = cpu_b.to(device)

        # Warming up
        _ = torch.matmul(gpu_a[:10, :10], gpu_b[:10, :10])
        torch.cuda.synchronize()

        start_time = time.time()
        torch.matmul(gpu_a, gpu_b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"  🐌 CPU Time: {cpu_time:.4f}s")
        print(f"  🚀 GPU Time: {gpu_time:.4f}s")
        print(f"  📈 Speedup: {speedup:.1f}x")

        return speedup > 1.0

    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False


def main():
    """Main execution."""
    print("🚀" + "=" * 50)
    print("   AxilDrive GPU Quick Check")
    print("=" * 52)

    # Basic check
    results = check_gpu_basic()

    # Environment variable check
    check_gpu_environment()

    # Performance test
    perf_ok = quick_performance_test()

    # Overall assessment
    print("\n" + "=" * 52)
    print("📊 Summary")
    print("=" * 52)

    if results.get("cuda_available", False):
        if results.get("tensor_ops", False) and results.get("neural_ops", False):
            print("🎉 GPU Ready: AxilDrive can use GPU acceleration!")
            print(f"   Device: {results.get('gpu_name', 'Unknown')}")
            print(f"   Memory: {results.get('gpu_memory_gb', 0):.1f}GB")
            if perf_ok:
                print("   Performance: Good acceleration detected")
            return True
        print("⚠️  GPU Partially Ready: Basic detection OK but operations failed")
        return False
    print("📱 CPU Mode: No GPU acceleration available")
    print("   AxilDrive will run on CPU (slower but functional)")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
