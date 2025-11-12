# LeRobot 录制最佳实践

## ✅ 成功的录制要点

### 1. Episode时长
- **推荐**: 每个episode 3-8秒
- **最短**: 至少2秒（60帧）
- **最长**: 不超过15秒
- **你的情况**: Episode1(5.6s) ✓  Episode2(4.1s) ✓  Episode3(2s) ✓

### 2. 动作质量
✅ 流畅完成任务
✅ 速度适中（不要太快或太慢）
✅ 成功抓取到目标物体
✅ 避免碰撞或意外停顿

❌ 动作失败（没抓到）
❌ 撞击物体
❌ 动作不连贯
❌ 速度过快导致相机模糊

### 3. 环境一致性
✅ 光照稳定
✅ 相机位置固定
✅ 背景简洁
✅ 物体起始位置相似

❌ 改变灯光
❌ 移动相机
❌ 杂乱背景
❌ 起始位置差异太大

### 4. 数据变化性
在保持环境一致的前提下，适当增加变化：
- 物体位置: 在工作区内5个不同位置
- 抓取角度: 不同的接近角度
- 抓取方式: 顶部抓/侧面抓

## 🔢 推荐的数据量

### 最小数据集
- **Episodes**: 50个
- **每个位置**: 10个episodes
- **变化点**: 5个不同起始位置

### 良好数据集  
- **Episodes**: 100个
- **每个位置**: 20个episodes
- **变化点**: 5个不同起始位置

### 优秀数据集
- **Episodes**: 200+个
- **更多变化**: 位置、角度、光照

## 📋 录制检查清单

### 每次录制前
- [ ] 机械臂校准正常
- [ ] 3个相机工作正常
- [ ] 工作台面整洁
- [ ] 光照充足稳定
- [ ] 目标物体准备好

### 录制过程中
- [ ] 动作流畅完成
- [ ] 没有人手进入镜头
- [ ] 相机视野清晰
- [ ] 任务成功完成

### 每个Episode后
- [ ] 检查保存是否成功
- [ ] 调整物体位置
- [ ] 准备下一个episode

## 🎬 录制命令模板

### 基础录制（5个episodes）
```bash
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
  --dataset.repo_id=myuser/grab_cube \
  --dataset.num_episodes=5 \
  --dataset.single_task="Grab the black cube" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30
```

### 生产级录制（50个episodes）
```bash
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
  --dataset.repo_id=myuser/grab_cube_v1 \
  --dataset.num_episodes=50 \
  --dataset.single_task="Grab the black cube" \
  --dataset.push_to_hub=true \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30
```

## 💡 常见问题

### Q: Episode太短会有问题吗？
A: 至少需要60帧（2秒@30fps）。你的Episode3只有62帧，刚好够。

### Q: 可以中途停止吗？
A: 按ESC键停止，已录制的episodes会保存。

### Q: 失败的尝试要删除吗？
A: 建议删除失败的episode，保持数据集质量。

### Q: 右箭头键不工作？
A: 确保终端窗口获得焦点，不要在其他窗口操作。

### Q: 录制时可以改变相机吗？
A: 不行！所有episodes必须使用相同的相机配置。
