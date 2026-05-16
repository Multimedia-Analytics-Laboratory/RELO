import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.utils.box_ops import box_xyxy_to_cxcywh

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))




class ValueCtrPredictor(nn.Module):
    def __init__(self, inplanes=64, channel=256, freeze_bn=False):
        super(ValueCtrPredictor, self).__init__()

        self.conv1_val = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_val = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_val = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_val = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)

        self.fc_val = nn.Linear(channel // 8, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass: input x from backbone, output V(s) """
        x_val = self.conv1_val(x)
        x_val = self.conv2_val(x_val)
        x_val = self.conv3_val(x_val)
        x_val = self.conv4_val(x_val)

        x_val = F.adaptive_avg_pool2d(x_val, 1).view(x_val.size(0), -1)
        value = self.fc_val(x_val)
        return value.squeeze(-1)


class ValueMlpModel(nn.Module):
    def __init__(self, inplanes=64, channel=256, hidden_dim=256):
        """
        in_channels: backbone output channels
        hidden_dim: hidden dimension
        """
        super(ValueMlpModel, self).__init__()
        self.fc1 = nn.Linear(inplanes, channel)
        self.fc2 = nn.Linear(channel, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        x: backbone features, shape [B, C, H, W]
        """
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value.squeeze(-1)



def build_value_model(cfg, encoder):
    in_channel = encoder.num_channels
    out_channel = cfg.MODEL.DECODER.NUM_CHANNELS
    type = cfg.MODEL.VALUE.TYPE
    if type == "CENTER":
        value_model = ValueCtrPredictor(inplanes=in_channel, channel=out_channel)
    elif type == "MLP":
        value_model = ValueMlpModel(inplanes=in_channel, channel=out_channel)
    else:
        raise NotImplementedError
    return value_model
