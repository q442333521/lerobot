# 🎯 LeRobot录制完整指南

## 📝 你的录制情况

### ✅ 已完成
- [x] 录制了3个episodes
- [x] 使用了3个相机视角
- [x] 数据成功保存
- [x] 视频编码完成

### 📊 数据详情
```
数据集: seeedstudio123/test
Episodes: 3个
总帧数: 354帧
总时长: 11.8秒
数据大小: 28.9 MB
```

## 🎬 正确的录制流程

### 一个完整的Episode

```
步骤1: 准备环境
┌──────────────────────────────┐
│ "Reset the environment"      │
│                              │
│ 要做:                        │
│ 1. 将黑色立方体放回起始位置  │
│ 2. 调整follower臂起始姿态    │
│ 3. 确保视野清晰              │
│                              │
│ 按键: [→] 右箭头键 开始录制 │
└──────────────────────────────┘
         ↓
步骤2: 执行动作
┌──────────────────────────────┐
│ "Recording episode X"        │
│                              │
│ 要做:                        │
│ 1. 用leader臂控制follower    │
│ 2. 流畅地抓取立方体          │
│ 3. 完成任务目标              │
│                              │
│ 按键: [→] 右箭头键 完成录制 │
│       (或等30秒自动结束)      │
└──────────────────────────────┘
         ↓
步骤3: 自动保存
┌──────────────────────────────┐
│ "Map: 100%|███| XXX/XXX"    │
│                              │
│ 系统自动:                    │
│ 1. 编码3个视频文件           │
│ 2. 保存机器人状态数据        │
│ 3. 更新元数据                │
│                              │
│ 等待: 几秒钟                 │
└──────────────────────────────┘
         ↓
    返回步骤1 (下一个Episode)
```

## ⌨️ 按键操作

| 按键 | 何时按 | 作用 |
|------|--------|------|
| **→** | 准备阶段 | 开始录制 |
| **→** | 录制阶段 | 完成episode |
| **ESC** | 任何时候 | 停止整个程序 |

## 📋 录制质量标准

### ✅ 好的Episode特征
- 动作流畅，无停顿
- 成功完成任务
- 时长3-8秒
- 相机视野清晰
- 无人手进入镜头

### ❌ 需要重录的情况
- 任务失败（没抓到）
- 撞击或意外停止
- 时长<2秒（太短）
- 相机被遮挡
- 环境突然变化

## 🔢 数据量建议

### 你目前的状态
```
当前: 3 episodes ❌ 数据太少
```

### 推荐数据量

#### 测试/调试
```
Episodes: 10-20个
目的: 验证流程
```

#### 最小可训练
```
Episodes: 50个
变化: 5个不同位置 × 10次
成功率: >80%能完成任务
```

####良好效果
```
Episodes: 100个
变化: 5个位置 × 20次
成功率: >90%
```

#### 优秀效果
```
Episodes: 200+个
变化: 更多位置、角度、方式
成功率: >95%
```

## 🎯 下一步操作

### 1. 查看你的数据
```bash
cd ~/lerobot
./view_dataset.sh
```

### 2. 继续录制更多数据
```bash
# 录制50个episodes
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ 
    top: {type: opencv, index_or_path: /dev/video1, width: 640, height: 480, fps: 30},
    front: {type: opencv, index_or_path: /dev/video11, width: 640, height: 480, fps: 30},
    wrist: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM3 \
  --dataset.repo_id=seeedstudio123/grab_cube_v1 \
  --dataset.num_episodes=50 \
  --dataset.single_task="Grab the black cube" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30
```

### 3. 训练策略模型 (至少50个episodes后)
```bash
lerobot-train \
  --dataset.repo_id=seeedstudio123/grab_cube_v1 \
  --policy.type=act \
  --output_dir=outputs/train/act_grab_cube \
  --job_name=act_grab_cube \
  --policy.device=cuda \
  --wandb.enable=false
```

## 💡 常见问题

### Q: 为什么我只录了3个就停了？
A: 你按了ESC键，这会停止整个录制程序。
   如果想完成当前episode，应该按→键。

### Q: Episode太短会影响训练吗？
A: 你的Episode3只有62帧(2秒)，刚好达标。
   建议保持3-8秒更好。

### Q: 可以追加更多episodes吗？
A: 可以！再次运行record命令，会追加到同一数据集。

### Q: 如何删除失败的episode？
A: 目前需要手动删除视频文件和元数据。
   建议完整重新录制一个新数据集。

### Q: 上传到HuggingFace有什么好处？
A: - 云端备份
   - 方便分享
   - 社区可以学习
   但初期测试建议 push_to_hub=false

## 📚 相关文档

- 录制技巧: ~/lerobot/RECORDING_TIPS.md
- 录制指南: ~/lerobot/RECORDING_GUIDE.md (未完成)
- 可视化脚本: ~/lerobot/view_dataset.sh

祝录制顺利！ 🚀
