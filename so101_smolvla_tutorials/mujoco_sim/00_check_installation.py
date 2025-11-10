#!/usr/bin/env python3
"""
安装检查脚本

这个脚本检查所有必需的依赖是否正确安装。

使用方法:
    python 00_check_installation.py
"""

import sys


def check_module(module_name, package_name=None, import_test=None):
    """检查模块是否安装"""
    if package_name is None:
        package_name = module_name

    try:
        if import_test:
            exec(import_test)
        else:
            __import__(module_name)
        print(f"✓ {package_name:20s} - 已安装")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} - 未安装")
        print(f"  错误: {e}")
        return False


def main():
    print("=" * 70)
    print("🔍 LeRobot MuJoCo 环境安装检查")
    print("=" * 70)
    print()

    all_ok = True

    # 基础依赖
    print("📦 基础依赖:")
    all_ok &= check_module("numpy", "NumPy")
    all_ok &= check_module("torch", "PyTorch")
    all_ok &= check_module("PIL", "Pillow")
    print()

    # LeRobot
    print("🤖 LeRobot:")
    all_ok &= check_module("lerobot", "LeRobot")
    all_ok &= check_module("lerobot.datasets", "LeRobot Datasets")
    all_ok &= check_module("lerobot.rl", "LeRobot RL")
    print()

    # MuJoCo 和 Gymnasium
    print("🎮 仿真环境:")
    all_ok &= check_module("mujoco", "MuJoCo")
    all_ok &= check_module("gymnasium", "Gymnasium")
    gym_hil_ok = check_module("gym_hil", "gym_hil")
    all_ok &= gym_hil_ok
    print()

    # SmolVLA
    print("🧠 SmolVLA:")
    smolvla_ok = check_module(
        "lerobot.policies.smolvla",
        "SmolVLA Policy",
        "from lerobot.policies.smolvla import SmolVLAPolicy"
    )
    all_ok &= smolvla_ok
    print()

    # GPU 检查
    print("💻 硬件:")
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ CUDA GPU           - {gpu_name}")
        print(f"  显存: {gpu_memory:.1f} GB")
    else:
        print("✗ CUDA GPU           - 未检测到")
        print("  警告: 没有 GPU，训练会非常慢")
    print()

    # 可选依赖
    print("🔧 可选依赖:")
    check_module("wandb", "Weights & Biases")
    check_module("cv2", "OpenCV", "import cv2")
    print()

    # 总结
    print("=" * 70)
    if all_ok:
        print("✅ 所有必需依赖都已安装！")
        print("\n🎉 您可以开始使用 MuJoCo 环境了")
        print("\n下一步:")
        print("  1. 配置 mujoco_so101_config.json")
        print("  2. 运行 python 01_collect_mujoco_data.py")
    else:
        print("❌ 部分依赖缺失")
        print("\n安装缺失的依赖:")

        if not gym_hil_ok:
            print("\n  gym_hil:")
            print("    pip install -e '.[hilserl]'")

        if not smolvla_ok:
            print("\n  SmolVLA:")
            print("    pip install -e '.[smolvla]'")

        print("\n或安装所有依赖:")
        print("  pip install -e '.[hilserl,smolvla]'")

    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
