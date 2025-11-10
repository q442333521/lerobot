# MuJoCo + SO101 + SmolVLA 完整教程

这个目录包含了在 MuJoCo 仿真环境中使用 SmolVLA 模型进行数据采集、训练和评估的完整工作流程。

## ⚠️ 重要说明：Sim2Real Gap

**MuJoCo 训练的模型不能直接用于实体 SO101 机械臂！**

仿真和真实环境存在显著差异：
- **物理差异**: 摩擦力、惯性、延迟
- **视觉差异**: 渲染 vs 真实摄像头
- **环境差异**: 光照、背景、物体材质

### MuJoCo 的正确用途

✅ **适合**:
- 快速验证算法流程
- 学习 LeRobot 和 SmolVLA 使用方法
- 在没有实体机械臂时进行学习
- 算法原型测试

❌ **不适合**:
- 直接部署到实体机械臂
- 需要精确物理交互的任务
- 生产环境使用

### 推荐工作流

```
MuJoCo 仿真 → 验证算法 → 真实机械臂小规模测试 → 大规模数据采集 → 最终部署
```

## 📋 目录结构

```
mujoco_sim/
├── README.md                          # 本文件
├── mujoco_so101_config.json          # 环境配置文件
├── mujoco_smolvla_quickstart.ipynb   # 快速入门教程 (概念)
├── 01_collect_mujoco_data.py         # 数据采集脚本
├── 02_train_smolvla.py               # 训练脚本
└── 03_evaluate_policy.py             # 评估脚本
```

## 🚀 快速开始

### 0. 前置要求

#### 硬件要求

**数据采集**:
- 游戏手柄 (Xbox/PS4 手柄)
- 或键盘控制

**训练**:
- GPU 推荐 (A100 / RTX 3090+)
- 16GB+ 显存
- 100GB+ 磁盘空间

#### 软件要求

```bash
# 1. 确保已安装 LeRobot
cd /path/to/lerobot
pip install -e .

# 2. 安装 MuJoCo 和 gym_hil
pip install -e ".[hilserl]"

# 3. 安装 SmolVLA 依赖
pip install -e ".[smolvla]"

# 4. 验证安装
python -c "import gym_hil; import mujoco; print('✓ 安装成功')"
```

### 1. 配置环境

编辑 `mujoco_so101_config.json`:

```json
{
  "dataset": {
    "repo_id": "your_username/mujoco_so101_pickplace",  // 修改为您的用户名
    "task": "Pick up the cube and place it on the target.",
    "num_episodes_to_record": 50
  }
}
```

### 2. 数据采集

#### 方法 A: 使用游戏手柄 (推荐)

```bash
# 连接游戏手柄后运行
python 01_collect_mujoco_data.py --config_path mujoco_so101_config.json

# 或指定参数
python 01_collect_mujoco_data.py \
  --config_path mujoco_so101_config.json \
  --num_episodes 50 \
  --push_to_hub
```

**控制说明**:
- **左摇杆**: X-Y 平面移动
- **右摇杆**: Z 轴移动和旋转
- **触发器**: 夹爪控制
- **Start**: 结束当前 episode
- **Select**: 重新录制当前 episode

#### 方法 B: 使用键盘

修改配置文件中的任务:
```json
"task": "PandaPickCubeKeyboard-v0"
```

然后运行相同的命令。

**键盘控制**:
- **WASD**: X-Y 平面移动
- **RF**: Z 轴上下
- **方向键**: 旋转
- **空格**: 夹爪
- **ESC**: 结束 episode

#### 数据质量建议

- 至少录制 **50 个** episodes
- 每个 episode 应包含**完整的抓取-放置过程**
- 尽量保持**动作流畅自然**
- 通过改变物体位置**增加多样性**
- 避免异常卡顿和失败的演示

### 3. 训练 SmolVLA 模型

```bash
python 02_train_smolvla.py \
  --dataset_repo_id your_username/mujoco_so101_pickplace \
  --batch_size 64 \
  --steps 20000 \
  --use_wandb
```

**参数说明**:
- `--dataset_repo_id`: 您的数据集 repo_id
- `--batch_size`: 批次大小 (根据 GPU 调整)
  - A100: 64
  - RTX 3090: 32-48
  - RTX 3080: 24-32
- `--steps`: 训练步数 (20000 约 4 小时)
- `--use_wandb`: 使用 Weights & Biases 监控

**预计训练时间**:
- A100: ~4 小时
- RTX 3090: ~6-8 小时
- CPU: 不推荐 (太慢)

**监控训练**:

如果启用了 W&B:
```
https://wandb.ai/your_username/lerobot-mujoco
```

### 4. 评估模型

```bash
python 03_evaluate_policy.py \
  --model_path outputs/mujoco_smolvla/checkpoint-20000 \
  --num_episodes 10 \
  --render \
  --save_results
```

**参数说明**:
- `--model_path`: 训练好的模型路径
- `--num_episodes`: 评估 episodes 数量
- `--render`: 显示环境渲染
- `--save_video`: 保存评估视频
- `--save_results`: 保存结果到 JSON

**评估指标**:
- 平均奖励
- 成功率
- Episode 长度

## 📊 完整工作流示例

```bash
# 步骤 1: 采集数据 (约 1-2 小时)
python 01_collect_mujoco_data.py \
  --config_path mujoco_so101_config.json \
  --num_episodes 50 \
  --push_to_hub

# 步骤 2: 训练模型 (约 4-8 小时)
python 02_train_smolvla.py \
  --dataset_repo_id your_username/mujoco_so101_pickplace \
  --batch_size 64 \
  --steps 20000 \
  --use_wandb

# 步骤 3: 评估模型 (约 10-20 分钟)
python 03_evaluate_policy.py \
  --model_path outputs/mujoco_smolvla/checkpoint-20000 \
  --num_episodes 10 \
  --render \
  --save_results
```

## 🔧 高级配置

### 自定义环境参数

编辑 `mujoco_so101_config.json`:

```json
{
  "env": {
    "fps": 20,  // 控制频率
    "processor": {
      "reset": {
        "control_time_s": 30.0  // Episode 最大时长
      },
      "gripper": {
        "gripper_penalty": -0.02  // 夹爪惩罚
      },
      "inverse_kinematics": {
        "end_effector_step_sizes": {
          "x": 0.025,  // X 轴步长
          "y": 0.025,  // Y 轴步长
          "z": 0.025   // Z 轴步长
        }
      }
    }
  }
}
```

### 训练超参数调优

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| batch_size | 64 | 32-128 | 根据 GPU 显存调整 |
| learning_rate | 1e-4 | 1e-5 ~ 5e-4 | 学习率 |
| steps | 20000 | 10000-50000 | 训练步数 |

### 使用预训练模型

```bash
python 02_train_smolvla.py \
  --policy_path lerobot/smolvla_base \  # 或其他预训练模型
  --dataset_repo_id your_username/mujoco_so101_pickplace
```

## 🐛 常见问题

### 1. 没有检测到游戏手柄

**问题**: 运行数据采集时提示找不到手柄

**解决**:
```bash
# Linux: 检查手柄设备
ls /dev/input/js*

# 测试手柄输入
jstest /dev/input/js0

# 安装驱动 (Ubuntu)
sudo apt-get install joystick
```

### 2. GPU 显存不足

**问题**: 训练时显存溢出

**解决**:
```bash
# 减小 batch size
python 02_train_smolvla.py --batch_size 32

# 使用梯度累积
python 02_train_smolvla.py --gradient_accumulation_steps 2
```

### 3. gym_hil 导入错误

**问题**: `ImportError: No module named 'gym_hil'`

**解决**:
```bash
# 重新安装
pip install -e ".[hilserl]"

# 验证
python -c "import gym_hil; print('OK')"
```

### 4. MuJoCo 许可证问题

**问题**: MuJoCo 2.x 及以上版本已免费，无需许可证

**解决**:
```bash
# 确保使用最新版本
pip install mujoco>=3.0.0
```

### 5. 数据集上传失败

**问题**: 上传到 Hugging Face Hub 失败

**解决**:
```bash
# 登录 HF
huggingface-cli login

# 检查仓库权限
# 确保 repo_id 格式正确: username/dataset_name
```

## 📚 参考资源

### LeRobot 官方文档

- **入门指南**: https://huggingface.co/docs/lerobot
- **SmolVLA 文档**: https://huggingface.co/docs/lerobot/smolvla
- **gym_hil 教程**: https://huggingface.co/docs/lerobot/hilserl_sim

### 相关项目

- **lerobot-mujoco**: https://github.com/q442333521/lerobot-mujoco
  - 更完整的 MuJoCo 教程
  - 包含 8 步详细指南
  - SO101 URDF/MJCF 模型

### 学术论文

**SmolVLA**:
```
@article{smolvla2024,
  title={SmolVLA: A Small Vision-Language-Action Model for Robotic Manipulation},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

**HIL-SERL**:
```
@article{luo2024precise,
  title={Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning},
  author={Luo, Jianlan and Xu, Charles and Wu, Jeffrey and Levine, Sergey},
  journal={arXiv preprint arXiv:2410.21845},
  year={2024}
}
```

## 💡 最佳实践

### 数据采集

1. **环境准备**:
   - 确保光照稳定
   - 背景简洁
   - 物体放置清晰可见

2. **演示质量**:
   - 动作平滑连贯
   - 每个 episode 包含完整任务
   - 避免重复失败的演示

3. **数据多样性**:
   - 改变物体位置
   - 不同起始姿态
   - 多种抓取方式

### 训练

1. **超参数选择**:
   - 从默认参数开始
   - 根据 loss 曲线调整
   - 使用 W&B 监控

2. **Checkpoint 管理**:
   - 定期保存 checkpoint
   - 保留多个版本
   - 评估选择最佳模型

3. **训练监控**:
   - 观察 loss 下降趋势
   - 检查过拟合
   - 定期评估验证集

### 评估

1. **多次评估**:
   - 至少 10 个 episodes
   - 计算平均性能
   - 统计成功率

2. **分析失败案例**:
   - 记录失败模式
   - 改进数据采集
   - 调整训练策略

## 🎓 下一步

完成 MuJoCo 训练后:

1. **迁移到真实机械臂**:
   - 参考 `../real_robot/` 目录
   - 采集真实数据
   - 微调或重新训练

2. **探索高级功能**:
   - 多任务学习
   - 域随机化
   - Sim2Real 技术

3. **社区贡献**:
   - 分享您的模型
   - 贡献改进
   - 帮助其他学习者

## 🤝 贡献与支持

- **问题反馈**: [GitHub Issues](https://github.com/huggingface/lerobot/issues)
- **讨论**: [Discord](https://discord.com/invite/s3KuuzsPFb)
- **文档**: [LeRobot Docs](https://huggingface.co/docs/lerobot)

---

**祝您学习愉快！🎉**

如有问题，欢迎在 LeRobot 社区寻求帮助。
