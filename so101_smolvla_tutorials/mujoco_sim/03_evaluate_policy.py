#!/usr/bin/env python3
"""
SmolVLA 策略评估脚本 (MuJoCo 环境)

这个脚本在 MuJoCo 仿真环境中评估训练好的 SmolVLA 模型。

使用方法:
    python 03_evaluate_policy.py --model_path outputs/mujoco_smolvla/checkpoint-20000

功能:
    - 加载训练好的模型
    - 在 MuJoCo 环境中运行多个 episodes
    - 统计成功率和性能指标
    - 可选保存视频
"""

import argparse
import json
import logging
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from lerobot.rl.gym_manipulator import make_robot_env, make_processors, GymManipulatorConfig
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.rl.gym_manipulator import DatasetConfig
from lerobot.processor import TransitionKey, create_transition

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_policy(model_path: str, device: str):
    """加载训练好的策略模型"""
    from lerobot.policies.smolvla import SmolVLAPolicy

    logger.info(f"加载模型: {model_path}")
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy = policy.to(device)
    policy.eval()

    param_count = sum(p.numel() for p in policy.parameters()) / 1e6
    logger.info(f"✓ 模型加载成功")
    logger.info(f"  参数量: {param_count:.1f}M")
    logger.info(f"  设备: {device}")

    return policy


def evaluate_policy(
    policy,
    env,
    env_processor,
    action_processor,
    num_episodes: int = 10,
    max_steps: int = 200,
    render: bool = True,
    save_video: bool = False,
    video_dir: Path = None
):
    """评估策略性能"""

    logger.info(f"\n🧪 开始评估 ({num_episodes} episodes)")
    logger.info(f"每个 episode 最多 {max_steps} 步")
    logger.info("=" * 70)

    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    for episode_idx in range(num_episodes):
        logger.info(f"\n📍 Episode {episode_idx + 1}/{num_episodes}")

        # Reset environment
        obs, info = env.reset()
        env_processor.reset()
        action_processor.reset()

        # Create initial transition
        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)

        episode_reward = 0.0
        episode_step = 0
        done = False

        frames = [] if save_video else None

        while episode_step < max_steps and not done:
            # Get action from policy
            with torch.no_grad():
                # 从观测中选择动作
                obs_tensor = transition[TransitionKey.OBSERVATION]
                action = policy.select_action(obs_tensor)

            # Process action
            transition[TransitionKey.ACTION] = action
            transition[TransitionKey.OBSERVATION] = (
                env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
            )

            processed_transition = action_processor(transition)
            processed_action = processed_transition[TransitionKey.ACTION]

            # Execute action
            obs, reward, terminated, truncated, info = env.step(processed_action)
            done = terminated or truncated

            # Record
            episode_reward += reward
            episode_step += 1

            # Create new transition
            new_transition = create_transition(
                observation=obs,
                action=processed_action,
                reward=reward,
                done=terminated,
                truncated=truncated,
                info=info
            )
            transition = env_processor(new_transition)

            # Render
            if render:
                env.render()

            # Save frame for video
            if save_video and frames is not None:
                # 获取渲染的图像
                # 这里需要根据实际的 gym_hil 环境 API 调整
                pass

            # Progress
            if episode_step % 10 == 0:
                print(f"\r  Step {episode_step}/{max_steps}, Reward: {episode_reward:.2f}", end="", flush=True)

        print()  # 换行

        # Record episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)

        # Determine success (需要根据具体任务定义)
        # 这里假设 reward > 0 表示成功
        success = episode_reward > 0
        episode_successes.append(success)

        logger.info(f"  完成: {episode_step} 步, 奖励: {episode_reward:.2f}, 成功: {success}")

        # Save video if requested
        if save_video and frames and video_dir:
            video_path = video_dir / f"episode_{episode_idx:03d}.mp4"
            # 这里需要实现视频保存逻辑
            logger.info(f"  视频已保存: {video_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 评估结果汇总")
    logger.info("=" * 70)
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} 步")
    logger.info(f"成功率: {np.mean(episode_successes) * 100:.1f}% ({sum(episode_successes)}/{num_episodes})")
    logger.info(f"最高奖励: {np.max(episode_rewards):.2f}")
    logger.info(f"最低奖励: {np.min(episode_rewards):.2f}")
    logger.info("=" * 70)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "success_rate": float(np.mean(episode_successes)),
    }


def main():
    parser = argparse.ArgumentParser(description="评估 SmolVLA 策略")

    # 模型配置
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的模型路径"
    )

    # 环境配置
    parser.add_argument(
        "--config_path",
        type=str,
        default="mujoco_so101_config.json",
        help="环境配置文件路径"
    )

    # 评估参数
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="评估的 episodes 数量"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="每个 episode 的最大步数"
    )

    # 可视化
    parser.add_argument(
        "--render",
        action="store_true",
        help="是否渲染环境"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="是否保存视频"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="视频保存目录"
    )

    # 设备
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备"
    )

    # 结果保存
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="保存评估结果到 JSON"
    )

    args = parser.parse_args()

    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，切换到 CPU")
        args.device = "cpu"

    # 加载配置
    logger.info(f"加载环境配置: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)

    # 修改配置为评估模式
    config_dict["mode"] = None  # 不录制数据
    config_dict["device"] = args.device
    config_dict["env"]["processor"]["observation"]["display_cameras"] = args.render

    # 检查依赖
    try:
        import gym_hil
        logger.info("✓ gym_hil 已安装")
    except ImportError:
        logger.error("✗ gym_hil 未安装!")
        logger.error("请运行: pip install -e '.[hilserl]'")
        return

    # 构建配置对象
    env_config = HILSerlRobotEnvConfig(**config_dict['env'])
    dataset_config = DatasetConfig(**config_dict['dataset'])

    gym_config = GymManipulatorConfig(
        env=env_config,
        dataset=dataset_config,
        mode=config_dict['mode'],
        device=config_dict['device']
    )

    logger.info("\n🚀 初始化环境...")

    try:
        # 加载模型
        policy = load_policy(args.model_path, args.device)

        # 创建环境 (评估模式不需要 teleop)
        env, _ = make_robot_env(gym_config.env)
        logger.info("✓ 环境创建成功")

        # 创建处理器
        env_processor, action_processor = make_processors(
            env, None, gym_config.env, gym_config.device
        )
        logger.info("✓ 处理器创建成功")

        # 创建视频目录
        video_dir = None
        if args.save_video:
            video_dir = Path(args.video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ 视频将保存到: {video_dir}")

        # 确认开始
        input("\n按 Enter 开始评估...")

        # 运行评估
        results = evaluate_policy(
            policy=policy,
            env=env,
            env_processor=env_processor,
            action_processor=action_processor,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=args.render,
            save_video=args.save_video,
            video_dir=video_dir
        )

        # 保存结果
        if args.save_results:
            results_file = Path(args.model_path).parent / "eval_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n💾 结果已保存到: {results_file}")

        logger.info("\n✅ 评估完成!")

    except KeyboardInterrupt:
        logger.info("\n⚠️ 用户中断评估")
    except Exception as e:
        logger.error(f"\n❌ 发生错误: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()
            logger.info("环境已关闭")


if __name__ == "__main__":
    main()
