"""
Encoder modules: we use ITPN for the encoder.
"""

from torch import nn
from lib.utils.misc import is_main_process
from . import fastitpn as fastitpn_module
from . import itpn as oriitpn_module



class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in encoder.named_parameters():

            if not train_encoder:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze:
                    parameter.requires_grad_(False)

        self.body = encoder
        self.num_channels = num_channels

    def forward(self, template_list, search_list, template_anno_list, ttokens):
        xs, ttokens_update = self.body(template_list, search_list, template_anno_list, ttokens)
        return xs, ttokens_update


class Encoder(EncoderBase):
    """Fast-iTPN/original iTPN encoder used by RELO."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 pretrain_type: str,
                 search_size: int,
                 search_number: int,
                 template_size: int,
                 template_number: int,
                 open_layers: list,
                 cfg=None):
        if "fastitpn" in name.lower():
            encoder = getattr(fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=cfg.MODEL.ENCODER.DROP_PATH,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                ttoken_type=cfg.MODEL.TTOKEN.TYPE,
                ttoken_block_idx=cfg.MODEL.TTOKEN.BLOCK_IDX,
                ttoken_proj=cfg.MODEL.TTOKEN.PROJ,
                ttoken_num=cfg.MODEL.TTOKEN.NUM,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "oriitpn" in name.lower():
            encoder = getattr(oriitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_path_rate=cfg.MODEL.ENCODER.DROP_PATH,
                init_values=0.1,
                use_mean_pooling=True,
                ape=True,
                rpe=True,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE
            )
            if "itpnb" in name:
                num_channels = 512
            else:
                num_channels = 512
        else:
            raise ValueError(f"Unsupported RELO encoder: {name}")
        super().__init__(encoder, train_encoder, open_layers, num_channels)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER is False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                      cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.NUMBER,
                      cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.NUMBER,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
