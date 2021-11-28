import torch
import torch.nn as nn
import torch.nn.functional as F
from models.visual_model import build_enc_vis

from pytorch_pretrained_bert.modeling import BertModel

from models.language_model import build_enc_lang
from .grounding_transformer import build_grounding_transformer
from utils.box_utils import xywh2xyxy

class VGTR(nn.Module):
    def __init__(self, args):
        super(VGTR, self).__init__()
        hidden_dim = args.hidden_dim
        
        self.visumodel = build_enc_vis(args)
        self.textmodel = build_enc_lang(args)

        self.g_transformer = build_grounding_transformer(args)
        self.bbox_embed = MLP(hidden_dim*4, hidden_dim, 4, 3)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        visu_out, visu_pos = self.visumodel(img_data)
        
        text_out = self.textmodel(text_data)
        
        g_trans_output = self.g_transformer(text_src=text_out.tensors, img_src=visu_out.tensors, pos_embed=visu_pos)
        
        g_trans_output = g_trans_output.squeeze().permute(1, 0, 2).flatten(1)
        out = self.bbox_embed(g_trans_output).sigmoid()
        #print(">>> VGTR out: ", out[12])
        return out



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
