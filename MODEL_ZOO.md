# RELO Model Zoo

[中文说明](MODEL_ZOO_zh-CN.md)

Public checkpoints and Fast-iTPN encoder weights are hosted in the Hugging Face repository `xche32/RELO`. The downloader keeps the public asset paths shown below.

| Model | Warmup config | RELO config | Encoder weight | Warmup checkpoint | RELO checkpoint | Download command |
| --- | --- | --- | --- | --- | --- | --- |
| RELO-T256 | `relo_warmup_t256` | `relo_t256` | `pretrained/itpn/fast_itpn_tiny_1600e_1k.pt` | `checkpoints/train/relo_warmup/relo_warmup_t256/RELO_WARMUP_ep0090.pth.tar` | `checkpoints/train/relo/relo_t256/RELO_ep0090.pth.tar` | `python tools/download_checkpoints.py --variant t256` |
| RELO-B256 | `relo_warmup_b256` | `relo_b256` | `pretrained/itpn/fast_itpn_base_clipl_e1600.pt` | `checkpoints/train/relo_warmup/relo_warmup_b256/RELO_WARMUP_ep0090.pth.tar` | `checkpoints/train/relo/relo_b256/RELO_ep0090.pth.tar` | `python tools/download_checkpoints.py --variant b256` |
| RELO-L256 | `relo_warmup_l256` | `relo_l256` | `pretrained/itpn/fast_itpn_large_1600e_1k.pt` | `checkpoints/train/relo_warmup/relo_warmup_l256/RELO_WARMUP_ep0090.pth.tar` | `checkpoints/train/relo/relo_l256/RELO_ep0090.pth.tar` | `python tools/download_checkpoints.py --variant l256` |

Download all public model assets:

```bash
python tools/download_checkpoints.py --variant all
```
