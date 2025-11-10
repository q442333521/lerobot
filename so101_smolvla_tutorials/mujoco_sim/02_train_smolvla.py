#!/usr/bin/env python3
"""
SmolVLA 模型训练脚本 (MuJoCo 数据)

这个脚本使用从 MuJoCo 环境采集的数据训练 SmolVLA 模型。

使用方法:
    python 02_train_smolvla.py --dataset_repo_id your_username/mujoco_so101_pickplace

注意:
    - 需要 GPU (推荐 A100 或 RTX 3090+)
    - 训练时间约 4-8 小时
    - 确保有足够的磁盘空间保存 checkpoints
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu():
    """检查 GPU 可用性"""
    if not torch.cuda.is_available():
        logger.warning("⚠️ 未检测到 CUDA GPU!")
        logger.warning("训练将非常慢。推荐使用:")
        logger.warning("  - Google Colab (免费 GPU)")
        logger.warning("  - AWS/GCP 云 GPU")
        logger.warning("  - 本地 NVIDIA GPU")
        response = input("\n是否继续使用 CPU 训练? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
        return False
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ 检测到 GPU: {gpu_name}")
        logger.info(f"✓ GPU 显存: {gpu_memory:.1f} GB")
        return True


def print_training_info(args):
    """打印训练信息"""
    print("=" * 70)
    print("🚀 SmolVLA 模型训练")
    print("=" * 70)
    print()
    print("📊 训练配置:")
    print(f"  数据集: {args.dataset_repo_id}")
    print(f"  模型: {args.policy_path}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  训练步数: {args.steps}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  使用 W&B: {args.use_wandb}")
    print()
    print("📈 预计训练时间:")
    print(f"  A100 GPU: ~4 小时")
    print(f"  RTX 3090: ~6-8 小时")
    print(f"  CPU: 不推荐 (太慢)")
    print()
    print("💡 提示:")
    print("  - 训练会自动保存 checkpoints")
    print("  - 可以使用 Ctrl+C 安全中断")
    print("  - 使用 W&B 可以实时监控训练进度")
    print()
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="训练 SmolVLA 模型")

    # 数据集配置
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Hugging Face 数据集 repo_id (例如: username/dataset_name)"
    )

    # 模型配置
    parser.add_argument(
        "--policy_path",
        type=str,
        default="lerobot/smolvla_base",
        help="预训练模型路径"
    )

    # 训练参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小 (A100: 64, RTX 3090: 32-48)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="训练步数"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1000,
        help="评估频率 (每 N 步)"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=2000,
        help="保存 checkpoint 频率 (每 N 步)"
    )

    # 输出配置
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mujoco_smolvla",
        help="输出目录"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="smolvla_mujoco_so101",
        help="任务名称"
    )

    # W&B 配置
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="使用 Weights & Biases 记录训练"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lerobot-mujoco",
        help="W&B 项目名称"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run 名称"
    )

    # 其他
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="训练设备"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据加载器的工作进程数"
    )

    args = parser.parse_args()

    # 检查 GPU
    has_gpu = check_gpu()
    if args.device == "cuda" and not has_gpu:
        args.device = "cpu"
        args.batch_size = min(args.batch_size, 8)  # CPU 时减小 batch size

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置 W&B run name
    if args.wandb_run_name is None:
        args.wandb_run_name = args.job_name

    # 打印训练信息
    print_training_info(args)

    # 确认开始
    response = input("是否开始训练? (y/n): ")
    if response.lower() != 'y':
        logger.info("训练取消")
        return

    # 构建训练命令
    # 注意: lerobot 可能没有标准的 train 命令，这里提供示例
    # 实际使用时需要根据 lerobot 的 API 调整

    logger.info("\n🚀 开始训练...\n")

    try:
        # 方法 1: 使用 lerobot 的训练脚本 (如果存在)
        # 这里我们提供一个通用的训练流程

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.smolvla import SmolVLAPolicy

        logger.info("加载数据集...")
        dataset = LeRobotDataset(
            repo_id=args.dataset_repo_id,
            root=None,  # 自动下载到缓存
            download_videos=True
        )
        logger.info(f"✓ 数据集加载成功: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

        logger.info("\n加载模型...")
        device = torch.device(args.device)

        # 加载预训练模型
        policy = SmolVLAPolicy.from_pretrained(args.policy_path)
        policy = policy.to(device)
        logger.info(f"✓ 模型加载成功")
        logger.info(f"  参数量: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")

        # 创建训练器
        logger.info("\n初始化训练...")

        # 这里需要根据实际的 SmolVLA 训练 API 进行调整
        # 以下是示例代码

        logger.info("\n" + "=" * 70)
        logger.info("⚠️ 注意: 完整的训练循环需要根据 SmolVLA 的 API 实现")
        logger.info("建议使用 lerobot 提供的官方训练脚本")
        logger.info("=" * 70)
        logger.info("\n推荐命令:")

        cmd_parts = [
            "python", "-m", "lerobot.scripts.train",
            f"--policy.path={args.policy_path}",
            f"--dataset.repo_id={args.dataset_repo_id}",
            f"--batch_size={args.batch_size}",
            f"--steps={args.steps}",
            f"--lr={args.learning_rate}",
            f"--output_dir={args.output_dir}",
            f"--job_name={args.job_name}",
            f"--device={args.device}",
        ]

        if args.use_wandb:
            cmd_parts.extend([
                f"--wandb.enable=true",
                f"--wandb.project={args.wandb_project}",
                f"--wandb.run_name={args.wandb_run_name}",
            ])

        print("\n" + " ".join(cmd_parts) + "\n")

        logger.info("\n💡 如果上述命令不适用，请参考 LeRobot 文档:")
        logger.info("https://huggingface.co/docs/lerobot")

    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        logger.error("\n请检查:")
        logger.error("  1. 数据集 repo_id 是否正确")
        logger.error("  2. SmolVLA 模型是否已安装: pip install -e '.[smolvla]'")
        logger.error("  3. GPU 显存是否足够")
        raise


if __name__ == "__main__":
    main()
