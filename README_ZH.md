<p align="center">
  <img alt="LeRobot, Hugging Face 机器人学习库" src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/lerobot-logo-thumbnail.png" width="100%">
  <br/>
  <br/>
</p>

<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://dcbadge.vercel.app/api/server/C5P34WJ68S?style=flat)](https://discord.gg/s3KuuzsPFb)

</div>

<h2 align="center">
    <p><a href="https://huggingface.co/docs/lerobot/hope_jr">
        构建您自己的 HopeJR 机器人！</a></p>
</h2>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/hope_jr/hopejr.png"
    alt="HopeJR 机器人"
    title="HopeJR 机器人"
    width="60%"
  />

  <p><strong>认识 HopeJR —— 用于灵巧操作的人形机械臂和手！</strong></p>
  <p>使用外骨骼和手套进行控制，实现精确的手部动作。</p>
  <p>适合高级操作任务！ 🤖</p>

  <p><a href="https://huggingface.co/docs/lerobot/hope_jr">
      查看完整的 HopeJR 教程。</a></p>
</div>

<br/>

<h2 align="center">
    <p><a href="https://huggingface.co/docs/lerobot/so101">
        构建您自己的 SO-101 机器人！</a></p>
</h2>

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/so101/so101.webp" alt="SO-101 从动臂" title="SO-101 从动臂" width="90%"/></td>
      <td align="center"><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/so101/so101-leader.webp" alt="SO-101 主动臂" title="SO-101 主动臂" width="90%"/></td>
    </tr>
  </table>

  <p><strong>认识升级版的 SO100，即 SO-101 —— 每个机械臂仅需 €114！</strong></p>
  <p>在笔记本电脑上通过几个简单的动作即可在几分钟内完成训练。</p>
  <p>然后坐下来观看您的创作自主行动！ 🤯</p>

  <p><a href="https://huggingface.co/docs/lerobot/so101">
      查看完整的 SO-101 教程。</a></p>

  <p>想要更进一步？通过构建 LeKiwi 让您的 SO-101 移动起来！</p>
  <p>查看 <a href="https://huggingface.co/docs/lerobot/lekiwi">LeKiwi 教程</a>，让您的机器人在轮子上活起来。</p>

  <img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/lekiwi/kiwi.webp" alt="LeKiwi 移动机器人" title="LeKiwi 移动机器人" width="50%">
</div>

<br/>

<h3 align="center">
    <p>LeRobot：用于真实世界机器人的最先进 AI</p>
</h3>

---

🤗 LeRobot 旨在为 PyTorch 提供用于真实世界机器人的模型、数据集和工具。目标是降低机器人技术的准入门槛，使每个人都能通过共享数据集和预训练模型做出贡献并从中受益。

🤗 LeRobot 包含已被证明可以迁移到真实世界的最先进方法，重点关注模仿学习和强化学习。

🤗 LeRobot 已经提供了一组预训练模型、包含人类演示的数据集以及模拟环境，让您无需组装机器人即可开始使用。在接下来的几周内，计划增加对市面上最实惠、最强大的真实世界机器人的更多支持。

🤗 LeRobot 在 Hugging Face 社区页面上托管预训练模型和数据集：[huggingface.co/lerobot](https://huggingface.co/lerobot)

#### 模拟环境上的预训练模型示例

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/aloha_act.gif" width="100%" alt="ALOHA 环境上的 ACT 策略"/></td>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/simxarm_tdmpc.gif" width="100%" alt="SimXArm 环境上的 TDMPC 策略"/></td>
    <td><img src="https://raw.githubusercontent.com/huggingface/lerobot/main/media/gym/pusht_diffusion.gif" width="100%" alt="PushT 环境上的 Diffusion 策略"/></td>
  </tr>
  <tr>
    <td align="center">ALOHA 环境上的 ACT 策略</td>
    <td align="center">SimXArm 环境上的 TDMPC 策略</td>
    <td align="center">PushT 环境上的 Diffusion 策略</td>
  </tr>
</table>

## 安装

LeRobot 支持 Python 3.10+ 和 PyTorch 2.2+。

### 环境设置

创建一个 Python 3.10 虚拟环境并激活它，例如使用 [`miniforge`](https://conda-forge.org/download/)：

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

使用 `conda` 时，在您的环境中安装 `ffmpeg`：

```bash
conda install ffmpeg -c conda-forge
```

> **注意：** 这通常会为您的平台安装使用 `libsvtav1` 编码器编译的 `ffmpeg 7.X`。如果不支持 `libsvtav1`（使用 `ffmpeg -encoders` 检查支持的编码器），您可以：
>
> - _[在任何平台上]_ 使用以下命令显式安装 `ffmpeg 7.X`：
>
> ```bash
> conda install ffmpeg=7.1.1 -c conda-forge
> ```
>
> - _[仅限 Linux]_ 安装 [ffmpeg 构建依赖项](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) 并[从源码编译带有 libsvtav1 的 ffmpeg](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1)，并确保使用与您的安装对应的 ffmpeg 二进制文件（使用 `which ffmpeg`）。

### 安装 LeRobot 🤗

#### 从源码安装

首先，克隆仓库并进入目录：

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

然后，以可编辑模式安装该库。如果您计划为代码做出贡献，这很有用。

```bash
pip install -e .
```

> **注意：** 如果您遇到构建错误，可能需要安装其他依赖项（`cmake`、`build-essential` 和 `ffmpeg libs`）。在 Linux 上，运行：
> `sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev`。对于其他系统，请参阅：[编译 PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)

对于模拟，🤗 LeRobot 附带可以作为额外组件安装的 gymnasium 环境：

- [aloha](https://github.com/huggingface/gym-aloha)
- [xarm](https://github.com/huggingface/gym-xarm)
- [pusht](https://github.com/huggingface/gym-pusht)

例如，要安装带有 aloha 和 pusht 的 🤗 LeRobot，请使用：

```bash
pip install -e ".[aloha, pusht]"
```

### 从 PyPI 安装

**核心库：**
使用以下命令安装基础包：

```bash
pip install lerobot
```

_这仅安装默认依赖项。_

**额外功能：**
要安装其他功能，请使用以下命令之一：

```bash
pip install 'lerobot[all]'          # 所有可用功能
pip install 'lerobot[aloha,pusht]'  # 特定功能（Aloha 和 Pusht）
pip install 'lerobot[feetech]'      # Feetech 电机支持
```

_将 `[...]` 替换为您需要的功能。_

**可用标签：**
有关可选依赖项的完整列表，请参阅：
https://pypi.org/project/lerobot/

> [!NOTE]
> 对于 lerobot 0.4.0，如果您想安装 libero 或 pi 标签，您需要执行：`pip install "lerobot[pi,libero]@git+https://github.com/huggingface/lerobot.git"`。
>
> 这将在下一个补丁版本中解决

### Weights & Biases

要使用 [Weights and Biases](https://docs.wandb.ai/quickstart) 进行实验跟踪，请使用以下命令登录：

```bash
wandb login
```

（注意：您还需要在配置中启用 WandB。请参阅下文。）

### 可视化数据集

查看[示例 1](https://github.com/huggingface/lerobot/blob/main/examples/dataset/load_lerobot_dataset.py)，它说明了如何使用我们的数据集类，该类会自动从 Hugging Face hub 下载数据。

您还可以通过从命令行执行我们的脚本来本地可视化 hub 上数据集的片段：

```bash
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

或者使用 `root` 选项和 `--mode local` 从本地文件夹中的数据集进行可视化（在以下情况下，将在 `./my_local_data_dir/lerobot/pusht` 中搜索数据集）：

```bash
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --mode local \
    --episode-index 0
```

它将打开 `rerun.io` 并显示相机流、机器人状态和动作，如下所示：

https://github-production-user-asset-6210df.s3.amazonaws.com/4681518/328035972-fd46b787-b532-47e2-bb6f-fd536a55a7ed.mov?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240505%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240505T172924Z&X-Amz-Expires=300&X-Amz-Signature=d680b26c532eeaf80740f08af3320d22ad0b8a4e4da1bcc4f33142c15b509eda&X-Amz-SignedHeaders=host&actor_id=24889239&key_id=0&repo_id=748713144

我们的脚本还可以可视化存储在远程服务器上的数据集。有关更多说明，请参阅 `lerobot-dataset-viz --help`。

### `LeRobotDataset` 格式

`LeRobotDataset` 格式的数据集非常易于使用。它可以从 Hugging Face hub 上的仓库或本地文件夹加载，只需简单地使用例如 `dataset = LeRobotDataset("lerobot/aloha_static_coffee")`，并且可以像任何 Hugging Face 和 PyTorch 数据集一样进行索引。例如，`dataset[0]` 将从数据集中检索包含观察和动作的单个时间帧，作为准备好馈送到模型的 PyTorch 张量。

`LeRobotDataset` 的一个特点是，我们可以根据索引帧与它们的时间关系检索多个帧，而不是通过索引检索单个帧，方法是将 `delta_timestamps` 设置为相对于索引帧的相对时间列表。例如，使用 `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}`，对于给定的索引，可以检索 4 个帧：3 个"先前"帧分别在索引帧之前 1 秒、0.5 秒和 0.2 秒，以及索引帧本身（对应于 0 条目）。有关 `delta_timestamps` 的更多详细信息，请参阅示例 [1_load_lerobot_dataset.py](https://github.com/huggingface/lerobot/blob/main/examples/dataset/load_lerobot_dataset.py)。

在底层，`LeRobotDataset` 格式使用多种方式来序列化数据，如果您计划更密切地使用此格式，了解这些方式可能很有用。我们试图制作一种灵活但简单的数据集格式，涵盖强化学习和机器人技术中存在的大多数类型的特征和特性，无论是在模拟还是现实世界中，重点关注相机和机器人状态，但很容易扩展到其他类型的感官输入，只要它们可以用张量表示。

以下是使用 `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` 实例化的典型 `LeRobotDataset` 的重要细节和内部结构组织。确切的特征将因数据集而异，但主要方面不会改变：

```
dataset 属性：
  ├ hf_dataset：一个 Hugging Face 数据集（由 Arrow/parquet 支持）。典型特征示例：
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': mp4 视频的路径, 'timestamp' (float32): 视频中的时间戳}
  │  ├ observation.state (float32 列表): 机械臂关节的位置（例如）
  │  ... (更多观察)
  │  ├ action (float32 列表): 机械臂关节的目标位置（例如）
  │  ├ episode_index (int64): 此样本的片段索引
  │  ├ frame_index (int64): 此样本在片段中的帧索引；每个片段从 0 开始
  │  ├ timestamp (float32): 片段中的时间戳
  │  ├ next.done (bool): 指示片段结束；对于每个片段的最后一帧为 True
  │  └ index (int64): 整个数据集中的一般索引
  ├ meta：一个 LeRobotDatasetMetadata 对象，包含：
  │  ├ info：数据集元数据的字典
  │  │  ├ codebase_version (str): 用于跟踪创建数据集的代码库版本
  │  │  ├ fps (int): 数据集记录/同步的每秒帧数
  │  │  ├ features (dict): 数据集中包含的所有特征及其形状和类型
  │  │  ├ total_episodes (int): 数据集中的总片段数
  │  │  ├ total_frames (int): 数据集中的总帧数
  │  │  ├ robot_type (str): 用于记录的机器人类型
  │  │  ├ data_path (str): parquet 文件的可格式化字符串
  │  │  └ video_path (str): 视频文件的可格式化字符串（如果使用视频）
  │  ├ episodes：包含片段元数据的 DataFrame，列包括：
  │  │  ├ episode_index (int): 片段的索引
  │  │  ├ tasks (list): 此片段的任务列表
  │  │  ├ length (int): 此片段中的帧数
  │  │  ├ dataset_from_index (int): 此片段在数据集中的起始索引
  │  │  └ dataset_to_index (int): 此片段在数据集中的结束索引
  │  ├ stats：数据集中每个特征的统计信息（最大值、平均值、最小值、标准差）的字典，例如
  │  │  ├ observation.images.front_cam: {'max': 具有相同维数的张量（例如，对于图像为 `(c, 1, 1)`，对于状态为 `(c,)`），等等}
  │  │  └ ...
  │  └ tasks：包含任务信息的 DataFrame，任务名称作为索引，task_index 作为值
  ├ root (Path): 存储数据集的本地目录
  ├ image_transforms (Callable): 应用于视觉模态的可选图像转换
  └ delta_timestamps (dict): 用于时间查询的可选增量时间戳
```

`LeRobotDataset` 使用多种广泛使用的文件格式对其每个部分进行序列化，即：

- hf_dataset 使用 Hugging Face datasets 库序列化为 parquet
- 视频以 mp4 格式存储以节省空间
- 元数据以纯 json/jsonl 文件存储

数据集可以无缝地从 HuggingFace hub 上传/下载。要使用本地数据集，如果它不在默认的 `~/.cache/huggingface/lerobot` 位置，您可以使用 `root` 参数指定其位置。

#### 复现最先进（SOTA）结果

我们在我们的 [hub 页面](https://huggingface.co/lerobot)上提供了一些预训练策略，可以实现最先进的性能。
您可以通过从它们的运行中加载配置来复现它们的训练。只需运行：

```bash
lerobot-train --config_path=lerobot/diffusion_pusht
```

即可在 PushT 任务上复现 Diffusion Policy 的 SOTA 结果。

## 贡献

如果您想为 🤗 LeRobot 做出贡献，请查看我们的[贡献指南](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md)。

### 添加预训练策略

一旦您训练了一个策略，您可以使用看起来像 `${hf_user}/${repo_name}` 的 hub id 将其上传到 Hugging Face hub（例如 [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)）。

您首先需要找到位于实验目录内的检查点文件夹（例如 `outputs/train/2024-05-05/20-21-12_aloha_act_default/checkpoints/002500`）。其中有一个 `pretrained_model` 目录，应包含：

- `config.json`：策略配置的序列化版本（遵循策略的数据类配置）。
- `model.safetensors`：一组 `torch.nn.Module` 参数，以 [Hugging Face Safetensors](https://huggingface.co/docs/safetensors/index) 格式保存。
- `train_config.json`：包含用于训练的所有参数的合并配置。策略配置应与 `config.json` 完全匹配。这对于任何想要评估您的策略或用于可重现性的人都很有用。

要将这些上传到 hub，请运行以下命令：

```bash
huggingface-cli upload ${hf_user}/${repo_name} path/to/pretrained_model
```

有关其他人如何使用您的策略的示例，请参阅 [lerobot_eval.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py)。

### 致谢

- LeRobot 团队 🤗 构建了 SmolVLA [论文](https://arxiv.org/abs/2506.01844)，[博客](https://huggingface.co/blog/smolvla)。
- 感谢 Tony Zhao、Zipeng Fu 及其同事开源 ACT 策略、ALOHA 环境和数据集。我们的版本改编自 [ALOHA](https://tonyzhaozh.github.io/aloha) 和 [Mobile ALOHA](https://mobile-aloha.github.io)。
- 感谢 Cheng Chi、Zhenjia Xu 及其同事开源 Diffusion 策略、Pusht 环境和数据集，以及 UMI 数据集。我们的版本改编自 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu) 和 [UMI Gripper](https://umi-gripper.github.io)。
- 感谢 Nicklas Hansen、Yunhai Feng 及其同事开源 TDMPC 策略、Simxarm 环境和数据集。我们的版本改编自 [TDMPC](https://github.com/nicklashansen/tdmpc) 和 [FOWM](https://www.yunhaifeng.com/FOWM)。
- 感谢 Antonio Loquercio 和 Ashish Kumar 的早期支持。
- 感谢 [Seungjae (Jay) Lee](https://sjlee.cc/)、[Mahi Shafiullah](https://mahis.life/) 及其同事开源 [VQ-BeT](https://sjlee.cc/vq-bet/) 策略并帮助我们将代码库适配到我们的仓库。该策略改编自 [VQ-BeT 仓库](https://github.com/jayLEE0301/vq_bet_official)。

## 引用

如果您愿意，可以引用这项工作：

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=huggingface/lerobot&type=Timeline)](https://star-history.com/#huggingface/lerobot&Timeline)
