# RELO 模型库

[English Model Zoo](MODEL_ZOO.md)

公开 checkpoints 和 Fast-iTPN encoder 权重托管在 Hugging Face 仓库 `xche32/RELO`。下载器会保留下表所示的公开资产路径。

| 模型 | Warmup 配置 | RELO 配置 | Encoder 权重 | Warmup checkpoint | RELO checkpoint | 下载命令 |
| --- | --- | --- | --- | --- | --- | --- |
| RELO-T256 | `relo_warmup_t256` | `relo_t256` | `pretrained/itpn/fast_itpn_tiny_1600e_1k.pt` | `checkpoints/train/relo_warmup/relo_warmup_t256/RELO_WARMUP_ep0090.pth.tar` | `checkpoints/train/relo/relo_t256/RELO_ep0090.pth.tar` | `python tools/download_checkpoints.py --variant t256` |
| RELO-B256 | `relo_warmup_b256` | `relo_b256` | `pretrained/itpn/fast_itpn_base_clipl_e1600.pt` | `checkpoints/train/relo_warmup/relo_warmup_b256/RELO_WARMUP_ep0090.pth.tar` | `checkpoints/train/relo/relo_b256/RELO_ep0090.pth.tar` | `python tools/download_checkpoints.py --variant b256` |
| RELO-L256 | `relo_warmup_l256` | `relo_l256` | `pretrained/itpn/fast_itpn_large_1600e_1k.pt` | `checkpoints/train/relo_warmup/relo_warmup_l256/RELO_WARMUP_ep0090.pth.tar` | `checkpoints/train/relo/relo_l256/RELO_ep0090.pth.tar` | `python tools/download_checkpoints.py --variant l256` |

下载全部公开模型资产：

```bash
python tools/download_checkpoints.py --variant all
```
