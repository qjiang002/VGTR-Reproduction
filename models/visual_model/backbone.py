# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from torch import nn
from typing import Dict
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from typing import Dict, List
from models.visual_model.dconv import Attention2d, DynamicConv2d
from . import dconv_resnet

from utils.misc import NestedTensor, is_main_process

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


class BackboneBase(nn.Module):

    def __init__(self, name: str, backbone: nn.Module, num_channels: int, return_interm_layers: bool, use_dconv: bool):
        super().__init__()
        self.use_dconv = use_dconv
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1",
                             "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(
            backbone, use_dconv, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(
                m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 return_interm_layers: bool,
                 dilation: bool, pretrained: bool, 
                 use_dconv: bool = False, dconv_places: str='first_in_block', dconv_7x7: bool=False, lang_dim: int = 768, K: int=5):

        assert name in ('resnet50', 'resnet101')
        if use_dconv:
            backbone = getattr(dconv_resnet, name)(lang_dim, K, dconv_places, dconv_7x7)
            print("Using dynamic convolution in backbone ResNet!")
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
            # pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        num_channels = 2048
        super().__init__(name, backbone, num_channels, return_interm_layers, use_dconv)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, use_dconv: bool, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        self.use_dconv = use_dconv
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: NestedTensor):
        lang = x.lang
        x = x.tensors
        out = OrderedDict()
        for name, module in self.items():
            if self.use_dconv:
                x = module(x, lang)
            else:
                x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out



class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos



def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # train_backbone = args.lr_detr > 0
    return_interm_layers = False

    pretrained = not args.no_pre_trained_resnet

    use_dconv = args.use_dconv
    lang_dim = None
    if use_dconv:
        if args.model_enc_lang == 'bert':
            lang_dim = 768
        else:
            raise NotImplemented(
                "Dynamic convolution with non-bert model not supported")

    backbone = Backbone(args.backbone, return_interm_layers,
                        args.dilation, pretrained=pretrained, 
                        use_dconv=use_dconv, dconv_places=args.dconv_places, dconv_7x7=args.dconv_7x7, 
                        lang_dim=lang_dim, K=args.dconv_candidate_count)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
