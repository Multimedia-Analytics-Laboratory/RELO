# RELO：用于视觉目标跟踪的强化学习定位方法

[English README](README.md)

本仓库包含 RELO 的 ICML 公开版本。公开版训练和推理使用 `relo_warmup` 和 `relo` 两个实验系列。

论文：[arXiv:2605.07379](https://arxiv.org/abs/2605.07379)

## 安装

创建并激活 conda 环境：

```bash
conda create -n relo python=3.10 -y
conda activate relo
```

安装 Python 依赖：

```bash
bash install.sh
```

## 数据和本地路径

从仓库根目录生成本地路径文件：

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

根据 `lib/train/admin/local.py` 和 `lib/test/evaluation/local.py` 中生成的路径，将数据集放在 `./data` 下。如果您的数据集目录结构不同，请调整这些文件。

## 检查点

公开检查点和所需的 Fast-iTPN encoder 权重托管在 Hugging Face 的 `xche32/RELO`。使用以下命令下载全部公开资产：

```bash
python tools/download_checkpoints.py --variant all
```

如果只下载一个模型规模，请将 `all` 替换为 `t256`、`b256` 或 `l256`。资产路径见 `MODEL_ZOO_zh-CN.md`。

## 训练

训练使用 `torchrun`。公开 YAML 文件写入了论文中的训练日程：先进行 90 个 warmup epoch，每个 epoch 100000 个样本；再进行 90 个 RELO epoch，每个 epoch 2500 个样本，学习率在第 72 个 epoch 下降。

Warmup 训练示例：

```bash
torchrun --nproc_per_node=8 lib/train/run_training.py \
  --script relo_warmup \
  --config relo_warmup_b256 \
  --save_dir .
```

RELO 训练示例：

```bash
torchrun --nproc_per_node=8 lib/train/run_training.py \
  --script relo \
  --config relo_b256 \
  --save_dir .
```

如果训练 T256 或 L256，请将 `relo_warmup_b256` 和 `relo_b256` 替换为对应的公开配置名：`relo_warmup_t256`、`relo_t256`、`relo_warmup_l256` 或 `relo_l256`。RELO 训练要求 YAML 中对应的 warmup checkpoint 路径已经存在于 `checkpoints/train/relo_warmup/...` 下；warmup 训练要求对应的 Fast-iTPN encoder 权重已经存在于 `pretrained/itpn/...` 下。

## 评测

在 LaSOT 上运行 RELO-B256：

```bash
python tracking/test.py relo relo_b256 --dataset_name lasot --threads 8 --num_gpus 1
```

分析结果：

```bash
python tracking/analysis_results.py relo relo_b256 --dataset_name lasot
```

支持的 `dataset_name` 包括 `lasot`、`lasot_extension_subset`、`trackingnet`、`got10k_test`、`tnl2k`、`nfs` 和 `uav`。不同数据集的评测设置，例如 window factor 和模板更新行为，已经写入公开 YAML 配置中。

## 联系方式

- Xin Chen（邮箱：xche32@cityu.edu.hk）

## 引用

如果本工作对您的研究有所帮助，我们诚挚希望您考虑引用我们的论文。

```bibtex
@inproceedings{relo,
  title={{RELO}: Reinforcement Learning to Localize for Visual Object Tracking},
  author={Xin Chen and Chuanyu Sun and Jiao Xu and Houwen Peng and Dong Wang and Huchuan Lu and Kede Ma},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
