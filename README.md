# RELO: Reinforcement Learning to Localize for Visual Object Tracking

[中文说明](README_zh-CN.md)

This repository contains the public ICML release for RELO. Public training and inference use the `relo_warmup` and `relo` experiment families.

Paper: [arXiv:2605.07379](https://arxiv.org/abs/2605.07379)

## Installation

Create and activate a conda environment:

```bash
conda create -n relo python=3.10 -y
conda activate relo
```

Install the Python dependencies:

```bash
bash install.sh
```

## Data and Local Paths

Generate local path files from the repository root:

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

Place datasets under `./data` using the paths created in `lib/train/admin/local.py` and `lib/test/evaluation/local.py`. Adjust those files if your dataset layout differs.

## Checkpoints

The public checkpoints and required Fast-iTPN encoder weights are hosted at `xche32/RELO` on Hugging Face. Download all released assets with:

```bash
python tools/download_checkpoints.py --variant all
```

To download only one model size, replace `all` with `t256`, `b256`, or `l256`. See `MODEL_ZOO.md` for asset paths.

## Training

Training uses `torchrun`. The public YAML files encode the paper schedule: 90 warmup epochs with 100000 samples per epoch, then 90 RELO epochs with 2500 samples per epoch and LR drop at epoch 72.

Warmup training example:

```bash
torchrun --nproc_per_node=8 lib/train/run_training.py \
  --script relo_warmup \
  --config relo_warmup_b256 \
  --save_dir .
```

RELO training example:

```bash
torchrun --nproc_per_node=8 lib/train/run_training.py \
  --script relo \
  --config relo_b256 \
  --save_dir .
```

For T256 or L256 training, replace `relo_warmup_b256` and `relo_b256` with the matching public config names: `relo_warmup_t256`, `relo_t256`, `relo_warmup_l256`, or `relo_l256`. RELO training expects the matching warmup checkpoint path from the YAML to exist under `checkpoints/train/relo_warmup/...`, and warmup training expects the matching Fast-iTPN encoder weight under `pretrained/itpn/...`.

## Evaluation

Run RELO-B256 on LaSOT:

```bash
python tracking/test.py relo relo_b256 --dataset_name lasot --threads 8 --num_gpus 1
```

Analyze the results:

```bash
python tracking/analysis_results.py relo relo_b256 --dataset_name lasot
```

Supported `dataset_name` values include `lasot`, `lasot_extension_subset`, `trackingnet`, `got10k_test`, `tnl2k`, `nfs`, and `uav`. Dataset-specific evaluation settings such as window factors and template update behavior are encoded in the public YAML configs.

## Contact

- Xin Chen (email: xche32@cityu.edu.hk)

## Citation

If this work proves helpful to your research, we kindly ask that you consider citing our paper.

```bibtex
@inproceedings{relo,
  title={{RELO}: Reinforcement Learning to Localize for Visual Object Tracking},
  author={Xin Chen and Chuanyu Sun and Jiao Xu and Houwen Peng and Dong Wang and Huchuan Lu and Kede Ma},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
