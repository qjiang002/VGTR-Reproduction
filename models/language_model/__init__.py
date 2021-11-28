from .bert import build_bert
from .bilstm import build_bilstm
from .vgtr_language import build_vgtr_language

def build_enc_lang(args):
    if args.model_enc_lang == 'bert':
        return build_bert(args) 
    elif args.model_enc_lang == 'bilstm':
        return build_bilstm(args)
    elif args.model_enc_lang == 'vgtr':
        return build_vgtr_language(args)
    else:
        raise NotImplemented