import torchvision.models as models
import torch
import torch.nn.functional as F
from torch import nn
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)
from .backbone import build_backbone

class Resnet(nn.Module):
    def __init__(self, backbone, train_resnet, use_dconv):
        """ Initializes the model.
        Parameters:
            resnet: a `BackboneBase` object, or an object with similar return types for `BackboneBase.__forward__`
        """
        super().__init__()

        self.backbone = backbone
        self.use_dconv = use_dconv

        hidden_dim = backbone.num_channels

        if not train_resnet:
            for p in self.resnet.parameters():
                p.requires_grad_(False)

        self.num_channels = hidden_dim

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            if self.use_dconv:
                raise NotImplemented # If using DConv, must have NestedTensor as input
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        src, mask, _ = features[-1].decompose()
        assert mask is not None

        out = [mask.flatten(1), src.flatten(2).permute(2, 0, 1)]
             
        return out


def build_resnet(args):
    resnet = build_backbone(args)
    train_resnet = args.lr_resnet> 0
    use_dconv = args.use_dconv

    model = Resnet(
        resnet,
        train_resnet=train_resnet,
        use_dconv=use_dconv
    )
    return model