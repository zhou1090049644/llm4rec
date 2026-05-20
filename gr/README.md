***

# 基于 Qwen2 的生成式推荐 SFT 训练框架

本项目提供了一套专门针对生成式推荐（Generative Recommendation）任务的监督微调（SFT）解决方案，其底层核心采用 **Qwen2** 架构。项目的核心思路是将用户过去的交互行为转换成 Semantic ID（语义 ID）序列，并以此来训练大语言模型（LLM），使其具备预测用户下一步可能感兴趣的物品（Item）的能力。

## 📁 目录结构

```
.
├── run.sh                 # [入口程序] 负责环境搭建及触发训练流程
├── run_train.sh           # [启动模块] 初始化分布式训练环境 (支持 Gloo/TCP) 并拉起 Python 主程序
├── train_gr.py            # [核心脚本] 训练主入口，涵盖模型读取与 Trainer 对象的实例化
├── gr_train.json          # [配置中心] 统筹模型架构、数据处理及训练过程的超参数
├── arguments.py           # [参数声明] 构建模型、数据和训练参数的数据类 (dataclasses)
├── custom_dataset.py      # [数据加载] 以流式方式读取 JSON，并将其转化为适配模型的输入格式
├── utils.py               # [辅助组件] 提供加载 Qwen2 模型、解析 Token 映射字典等实用功能
└── requirements.txt       # 运行所需 Python 第三方库清单
```

## 🛠️ 运行环境

本工程的底层主要依赖于 **PyTorch**、**Transformers**、**DeepSpeed** 以及 **Accelerate** 构建。

直接执行 `run.sh` 即可自动完成核心依赖包的部署，主要环境要求如下：

- Python 3.8 或以上版本
- PyTorch (兼容 CUDA 11.7 或 12.x 系列)
- DeepSpeed == 0.15.4
- Transformers == 4.46.3
- HuggingFace Hub == 0.33.4
- MPI (OpenMPI 环境)

## 📊 数据与资源准备

在正式启动训练任务前，务必备齐初始化的模型权重与数据集，并将它们存放于特定的环境变量指向的目录中。

### 1. 环境变量配置

程序运行高度依赖以下环境变量来定位资源和输出结果：

- `USER_CACHE_PATH`: 存放预训练模型、基础训练数据以及映射字典的基础根目录。
- `TRAIN_CKPT_PATH`: 用于保存微调后输出的模型权重（Checkpoints）的路径。
- `RUNTIME_SCRIPT_DIR`: 当前执行脚本所在的路径（平台一般会自动分配，特殊情况下需人工指定）。

### 2. 预训练权重 (Qwen2 Init)

请仔细检查 `$USER_CACHE_PATH/qwen_init2` 目录下是否已完整存放 Qwen2 的初始权重及其配套文件（例如 `config.json` 和 `tokenizer.json` 等）。

### 3. ID 映射字典 (Item2Token)

由于生成式推荐依赖 Semantic ID，系统需要一份将实体 Item 转换为对应 Token 的映射表。

- **存放路径**: `$USER_CACHE_PATH/emb_infer/sinkhorn10`（若需变更，请在 `gr_train.json` 内修改 `item2token_dict` 字段）。
- **文件形态**: 包含具体文本文档的文件夹。
- **排版要求**: 逐行对应，格式为 `item_id \t token`（中间以 Tab 键隔开）。

### 4. 训练集格式

系统原生支持以 JSON 格式进行数据的流式读取。

- **默认文件命名**: `train_data.json`
- **默认读取路径**: `$USER_CACHE_PATH/train_data.json`
- **内部数据结构**:

  JSON

  ```
  {
      "user_id_1": ["item_id_A", "item_id_B", "item_id_C"],
      "user_id_2": ["item_id_X", "item_id_Y"]
  }
  ```

  *提示：系统会自动剔除序列总长度不足 2 的用户数据（因为必须保证至少有 1 个历史行为记录和 1 个预测目标）。*

## 🚀 启动训练流程

整个工程支持便捷的一键式启动操作：

Bash

```
bash run.sh
```

### 执行逻辑解析：

1. **装配环境**: `run.sh` 首先会解决 `pip` 和 `conda` 相关的依赖安装。
2. **分布式初始化**: `run_train.sh` 负责设定分布式训练的网络参数。
   - **重点提示**: 考虑到对部分非 RDMA 网络环境的兼容，当前脚本硬编码采用了 `Gloo` 作为后端，主动关闭了 `InfiniBand/RDMA` (`NCCL_IB_DISABLE=1`) 并强制启用 TCP 协议 (`DS_TRANSPORT_TCP=1`)。如果您的硬件支持 NVLink 或 RDMA，建议手动调整 `run_train.sh` 以释放最佳性能。
3. **触发训练**: 执行 `train_gr.py`，根据 `gr_train.json` 内的配置参数正式开启模型微调。

## ⚙️ 超参数调整 (`gr_train.json`)

各类核心控制参数均集中在 JSON 配置文件中进行管理：

| **参数分类** | **关键字段** | **功能描述** |
| :--- | :--- | :--- |
| **model_args** | `se_id_space_width` | 设定语义 ID 各层级的空间跨度（示例："2048,2048,1024"） |
| **data_args** | `max_seq_length` | 限制截断的用户行为序列最大长度（默认值：100） |
| | `token_depth` | 定义 Semantic ID 的层级维度（默认值：3） |
| **training_args**| `output_dir` | 设定最终模型保存的子目录名称（生成于 `$TRAIN_CKPT_PATH` 中）|
| | `learning_rate` | 全局初始学习率（默认值：1e-4） |
| | `per_device_train_batch_size` | 每个计算设备的 Batch Size |
| | `gradient_accumulation_steps` | 进行梯度更新前的累积步数 |

## 🧩 核心逻辑解析

- **模型输入重构 (custom_dataset.py)**:

  系统在数据加载阶段会按如下范式拼接 Prompt：

  Plaintext

  ```
  Input: <|hist_clk_start|> [Token_Item_1] [Token_Item_2] ... <|hist_clk_end|> [Target_Item_Token]
  Label: [Target_Item_Token] (模型前向传播时，仅针对 Target 部分的预测结果计算 Loss)
  ```

- **DeepSpeed 引擎**: 本项目通过 `transformers.Trainer` 实现了对 DeepSpeed 的原生兼容。尽管在 `run_train.sh` 里面主要是依靠环境变量来做手动声明，但底层代码已完全打通了 DeepSpeed 的接入逻辑。

## ⚠️ 常见排错指南

- **通信与网络故障**: 若触发了 NCCL 通信超时或断连报错，请核查 `run_train.sh` 内的 `NCCL_SOCKET_IFNAME=lo` 配置。该默认项主要适配本地单机回环调试，**如果在多机集群中部署，请务必将其替换为物理网卡的真实标识（例如 `eth0` 等）**。
- **文件寻址失败**: 遇到 `FileNotFoundError` 报错时，第一步请排查宿主机是否正确注入了 `USER_CACHE_PATH` 环境变量。