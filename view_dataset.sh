#!/bin/bash

# 查看数据集
lerobot-dataset-viz \
  --repo-id seeedstudio123/test \
  --mode local \
  --root ~/.cache/huggingface/lerobot \
  --episode-index 0

# 可以改变 --episode-index 查看不同的episode
# --episode-index 0  查看第1个episode
# --episode-index 1  查看第2个episode
# --episode-index 2  查看第3个episode
