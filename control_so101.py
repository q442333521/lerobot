#!/usr/bin/env python3
import torch, argparse
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

parser = argparse.ArgumentParser()
parser.add_argument("--policy_path", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--port", default="/dev/ttyACM0")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=100)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}, 任务: {args.task}")

model = SmolVLAPolicy.from_pretrained(args.policy_path).to(device)
preprocess, postprocess = make_pre_post_processors(model.config, args.policy_path, preprocessor_overrides={"device_processor": {"device": str(device)}})

camera_config = {"top": OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=30, fourcc="MJPG"), "front": OpenCVCameraConfig(index_or_path=8, width=640, height=480, fps=30, fourcc="MJPG"), "wrist": OpenCVCameraConfig(index_or_path=6, width=640, height=480, fps=30, fourcc="MJPG")}

robot_cfg = SO101FollowerConfig(port=args.port, id="my_awesome_follower_arm", cameras=camera_config)
robot = SO101Follower(robot_cfg)
robot.connect()
print("机器人已连接")

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

for ep in range(args.num_episodes):
    print(f"Episode {ep+1}/{args.num_episodes}")
    for step in range(args.max_steps):
        obs = robot.get_observation()
        obs_frame = build_inference_frame(observation=obs, ds_features=dataset_features, device=device, task=args.task, robot_type="so101_follower")
        obs = preprocess(obs_frame)
        action = model.select_action(obs)
        action = postprocess(action)
        action = make_robot_action(action, dataset_features)
        robot.send_action(action)
    print(f"Episode {ep+1} 完成")

robot.disconnect()
print("完成")
