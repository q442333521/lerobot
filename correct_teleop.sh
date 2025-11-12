#!/bin/bash

lerobot-teleoperate \
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
  --display_data=true

