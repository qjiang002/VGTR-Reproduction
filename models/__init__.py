from .trans_vg import TransVG
from .vgtr import VGTR

def build_model(args):
    if args.model_name=='TransVG':
        return TransVG(args)
    elif args.model_name=='vgtr':
        return VGTR(args)
    
    raise NotImplemented

