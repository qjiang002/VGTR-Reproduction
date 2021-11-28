from .detr import build_detr
from .resnet import build_resnet
from .vgtr_visual import build_vgtr_visual

def build_enc_vis(args):
    model = args.model_enc_vis
    if model == 'detr':
        return build_detr(args) 
    elif model == 'resnet':
        return build_resnet(args)
    elif model == 'vgtr':
        return build_vgtr_visual(args)

    raise NotImplemented