#!/bin/bash

# 设置HuggingFace用户名（可选，如果要上传）
export HF_USER=seeedstudio123

# 数据采集命令
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ 
      top: {
        type: opencv, 
        index_or_path: /dev/video1, 
        width: 640, 
        height: 480, 
        fps: 30
      },
      front: {
        type: opencv, 
        index_or_path: /dev/video11, 
        width: 640, 
        height: 480, 
        fps: 30
      },
      wrist: {
        type: opencv, 
        index_or_path: /dev/video6, 
        width: 640, 
        height: 480, 
        fps: 30
      }
    }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM3 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/grab_black_cube \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the black cube" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30
