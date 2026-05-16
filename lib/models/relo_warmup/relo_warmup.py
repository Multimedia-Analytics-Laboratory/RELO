import math
from torch import nn
from .encoder import build_encoder
from .decoder import build_decoder


class RELOWarmup(nn.Module):
    """Regression warmup model for RELO."""

    checkpoint_name = "RELO_WARMUP"
    checkpoint_aliases = ("RELOWarmup",)

    def __init__(self, encoder, decoder, decoder_type="CENTER"):
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder



    def forward(self, template_list=None, search_list=None, template_anno_list=None,
                ttokens=None, feature=None, mode="encoder"):
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, ttokens)
        elif mode == "decoder":
            return self.forward_decoder(feature)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list, template_anno_list, ttokens):
        # Forward the encoder
        xz, ttokens_update = self.encoder(template_list, search_list, template_anno_list, ttokens)
        return xz, ttokens_update

    def forward_decoder(self, feature, gt_score_map=None):
        feature = feature[0]
        feature = feature[:,0:self.num_patch_x] # (B, HW, C)
        bs, HW, C = feature.size()
        if self.decoder_type in ['CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_relo_warmup(cfg):
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg, encoder)
    model = RELOWarmup(
        encoder,
        decoder,
        decoder_type=cfg.MODEL.DECODER.TYPE
    )

    return model
