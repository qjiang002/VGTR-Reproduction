from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import torchvision.models.detection.backbone_utils as backbone_utils

from utils.misc import NestedTensor, is_main_process, nested_tensor_from_tensor_list
from .position_encoding import build_position_encoding

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
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):

    def __init__(self, name: str):
        super().__init__()
        assert name in ('resnet50', 'resnet101', 'resnet152')
        pre_backbone = backbone_utils.resnet_fpn_backbone(name, pretrained=True)
        features = list(pre_backbone.children())[:-1]
        self.backbone = nn.Sequential(*features)
        self.num_channels = [256, 512, 1024, 2048]

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class VisualModel(nn.Module):
    def __init__(self, resnet, train_resnet, out_dim):
        super().__init__()

        self.resnet = resnet

        if not train_resnet:
            for p in self.resnet.parameters():
                p.requires_grad_(False)

        self.input_proj_0 = nn.Conv2d(256, out_dim, kernel_size=8, stride=8)
        self.input_proj_1 = nn.Conv2d(512, out_dim, kernel_size=4, stride=4)
        self.input_proj_2 = nn.Conv2d(1024, out_dim, kernel_size=2, stride=2)
        self.input_proj_3 = nn.Conv2d(2048, out_dim, kernel_size=1, stride=1)
        self.visual_token_proj = nn.Conv2d(out_dim*4, out_dim, kernel_size=1)
        self.num_channels = out_dim

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.resnet(samples)
        
        proj_0 = self.input_proj_0(features['0'].tensors)
        proj_1 = self.input_proj_1(features['1'].tensors)
        proj_2 = self.input_proj_2(features['2'].tensors)
        proj_3 = self.input_proj_3(features['3'].tensors)
        
        concat_F = torch.cat([proj_0, proj_1, proj_2, proj_3], 1)
        visual_tokens = self.visual_token_proj(concat_F)
        #visual_tokens = visual_tokens.flatten(2).permute(0, 2, 1)
        #assert visual_tokens.shape[2] == self.num_channels
        mask = torch.empty_like(features['3'].mask).fill_(0).to(torch.bool)
        #print("mask: ", mask.shape)
        out = NestedTensor(visual_tokens, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, visual_model, position_embedding):
        super().__init__(visual_model, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        
        out:NestedTensor = self[0](tensor_list)
        pos = self[1](out).to(out.tensors.dtype)
        #print("pos: ", pos.shape)
        return out, pos

def build_vgtr_visual(args):
    
    backbone = Backbone(args.backbone)
    train_resnet = args.lr_resnet > 0
    position_embedding = build_position_encoding(args)
    visual_model = VisualModel(
        backbone,
        train_resnet=train_resnet,
        out_dim = args.vgtr_visual_out_dim
    )
    model = Joiner(visual_model, position_embedding)
    model.num_channels = visual_model.num_channels
    return model
    