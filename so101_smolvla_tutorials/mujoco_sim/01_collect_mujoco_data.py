#!/usr/bin/env python3
"""
MuJoCo SO101 数据采集脚本

这个脚本使用 MuJoCo 仿真环境采集训练数据，用于 SmolVLA 模型训练。

使用方法:
    python 01_collect_mujoco_data.py --config_path mujoco_so101_config.json

控制说明:
    - 使用游戏手柄控制机械臂
    - 左摇杆: X-Y 平面移动
    - 右摇杆: Z 轴移动和旋转
    - 触发器: 夹爪控制
    - Start: 结束当前 episode
    - Select: 重新录制当前 episode
"""

import argparse
import json
import logging
from pathlib import Path

import gymnasium as gym
import torch

from lerobot.rl.gym_manipulator import make_robot_env, make_processors, control_loop, GymManipulatorConfig
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import DatasetConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载 JSON 配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def print_instructions():
    """打印使用说明"""
    print("=" * 70)
    print("🎮 MuJoCo SO101 数据采集")
    print("=" * 70)
    print()
    print("📋 开始前检查清单:")
    print("  ✓ 游戏手柄已连接")
    print("  ✓ GPU 可用 (推荐)")
    print("  ✓ 配置文件已正确设置")
    print()
    print("🎮 控制说明:")
    print("  - 左摇杆: 控制 X-Y 平面移动")
    print("  - 右摇杆: 控制 Z 轴移动和旋转")
    print("  - 触发器: 夹爪开关")
    print("  - Start 按钮: 结束当前 episode")
    print("  - Select 按钮: 重新录制当前 episode")
    print()
    print("💡 提示:")
    print("  - 每个 episode 应该包含完整的抓取-放置过程")
    print("  - 尽量保持动作流畅自然")
    print("  - 录制至少 50 个 episodes 以获得好的训练效果")
    print("  - 可以通过改变物体位置增加数据多样性")
    print()
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo SO101 数据采集")
    parser.add_argument(
        "--config_path",
        type=str,
        default="mujoco_so101_config.json",
        help="配置文件路径"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="要录制的 episodes 数量 (覆盖配置文件)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="数据集保存目录 (覆盖配置文件)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="录制完成后上传到 Hugging Face Hub"
    )
    args = parser.parse_args()

    # 打印说明
    print_instructions()

    # 加载配置
    logger.info(f"加载配置文件: {args.config_path}")
    config_dict = load_config(args.config_path)

    # 覆盖命令行参数
    if args.num_episodes:
        config_dict["dataset"]["num_episodes_to_record"] = args.num_episodes
    if args.output_dir:
        config_dict["dataset"]["root"] = args.output_dir
    if args.push_to_hub:
        config_dict["dataset"]["push_to_hub"] = True

    # 确保 mode 设置为 record
    config_dict["mode"] = "record"

    # 打印配置信息
    logger.info("=" * 50)
    logger.info("配置信息:")
    logger.info(f"  环境: {config_dict['env']['task']}")
    logger.info(f"  FPS: {config_dict['env']['fps']}")
    logger.info(f"  数据集名称: {config_dict['dataset']['repo_id']}")
    logger.info(f"  任务描述: {config_dict['dataset']['task']}")
    logger.info(f"  Episodes 数量: {config_dict['dataset']['num_episodes_to_record']}")
    logger.info(f"  Episode 时长: {config_dict['env']['processor']['reset']['control_time_s']}s")
    logger.info(f"  保存路径: {config_dict['dataset'].get('root', '默认')}")
    logger.info(f"  上传到 Hub: {config_dict['dataset']['push_to_hub']}")
    logger.info(f"  设备: {config_dict['device']}")
    logger.info("=" * 50)

    # 检查依赖
    try:
        import gym_hil
        logger.info("✓ gym_hil 已安装")
    except ImportError:
        logger.error("✗ gym_hil 未安装!")
        logger.error("请运行: pip install -e '.[hilserl]'")
        return

    # 检查 CUDA
    if config_dict['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，切换到 CPU")
        config_dict['device'] = 'cpu'

    # 构建配置对象
    env_config = HILSerlRobotEnvConfig(**config_dict['env'])
    dataset_config = DatasetConfig(**config_dict['dataset'])

    gym_config = GymManipulatorConfig(
        env=env_config,
        dataset=dataset_config,
        mode=config_dict['mode'],
        device=config_dict['device']
    )

    logger.info("\n🚀 开始初始化环境...")

    try:
        # 创建环境
        env, teleop_device = make_robot_env(gym_config.env)
        logger.info("✓ 环境创建成功")

        # 创建处理器
        env_processor, action_processor = make_processors(
            env, teleop_device, gym_config.env, gym_config.device
        )
        logger.info("✓ 处理器创建成功")

        logger.info(f"\n观测空间: {env.observation_space}")
        logger.info(f"动作空间: {env.action_space}\n")

        # 确认开始
        input("按 Enter 开始录制...")

        # 开始数据采集
        logger.info("\n🎬 开始数据采集...\n")
        control_loop(env, env_processor, action_processor, teleop_device, gym_config)

        logger.info("\n✅ 数据采集完成!")

        if gym_config.dataset.push_to_hub:
            logger.info(f"📤 数据已上传到: https://huggingface.co/datasets/{gym_config.dataset.repo_id}")

    except KeyboardInterrupt:
        logger.info("\n⚠️ 用户中断数据采集")
    except Exception as e:
        logger.error(f"\n❌ 发生错误: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()
            logger.info("环境已关闭")


if __name__ == "__main__":
    main()
