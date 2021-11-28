
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class G_Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = _get_clones(nn.LayerNorm(d_model), 2) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if num_decoder_layers > 0:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)
        else:
            self.decoder = None

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, text_src, img_src, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = img_src.shape
        text_src = text_src.permute(1, 0, 2)
        img_src = img_src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        #img_mask = img_mask.flatten(1)
        t_output, v_output = self.encoder(text_src, img_src, pos=pos_embed)
        #print(">>> G_Transformer encoder t_output v_output: ", t_output.shape, v_output.shape)
        if self.decoder is not None:
            hs = self.decoder(t_output, v_output, pos=pos_embed)
            return hs
        else:
            return t_output, v_output
'''
class TransformerEncOnly(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        return memory.permute(1, 2, 0).view(bs, c, h, w)
'''

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, text_src, img_src,
                img_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        v_output = img_src
        t_output = text_src

        for layer in self.layers:
            t_output, v_output = layer(t_output, v_output, img_mask=img_mask,
                                    src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            t_output = self.norm[0](t_output)
            v_output = self.norm[1](v_output)

        return t_output, v_output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, t_output, v_output,
                img_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = t_output

        intermediate = []

        for layer in self.layers:
            output = layer(output, v_output, img_mask, src_key_padding_mask, pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
        super().__init__()
        self.v_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.t_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.v_linear1 = nn.Linear(d_model, dim_feedforward)
        self.v_linear_dropout = nn.Dropout(dropout)
        self.v_linear2 = nn.Linear(dim_feedforward, d_model)
        self.t_linear1 = nn.Linear(d_model, dim_feedforward)
        self.t_linear_dropout = nn.Dropout(dropout)
        self.t_linear2 = nn.Linear(dim_feedforward, d_model)

        self.v_norm1 = nn.LayerNorm(d_model)
        self.v_norm2 = nn.LayerNorm(d_model)
        self.t_norm1 = nn.LayerNorm(d_model)
        self.t_norm2 = nn.LayerNorm(d_model)
        self.v_dropout1 = nn.Dropout(dropout)
        self.v_dropout2 = nn.Dropout(dropout)
        self.t_dropout1 = nn.Dropout(dropout)
        self.t_dropout2 = nn.Dropout(dropout)

        self.v_activation = _get_activation_fn(activation)
        self.t_activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    '''
    def forward_post(self,
                     text_src,
                     img_src,
                     img_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=img_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    '''
    def forward_pre(self, text_src, img_src,
                    img_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        img_src2 = self.v_norm1(img_src)
        Qv = Kv = self.with_pos_embed(img_src2, pos)
        Vv = img_src2
        text_src2 = self.t_norm1(text_src)
        Qt = Kt = Vt = text_src2
        text_atten_output, text_atten_weight = self.t_self_attn(Qt, Kt, Vt)

        cross_atten_output, cross_atten_weight = self.cross_attn(Qv, text_atten_output, text_atten_output)

        Qv2 = Qv + cross_atten_output

        visual_atten_output, visual_atten_weight = self.v_self_attn(Qv2, Kv, Vv)#, attn_mask=img_mask, key_padding_mask=src_key_padding_mask)

        img_src = img_src + self.v_dropout1(visual_atten_output)
        text_src = text_src + self.t_dropout1(text_atten_output)

        img_src2 = self.v_norm2(img_src)
        text_src2 = self.t_norm2(text_src)
        img_src2 = self.v_linear2(self.v_linear_dropout(self.v_activation(self.v_linear1(img_src2))))
        text_src2 = self.t_linear2(self.t_linear_dropout(self.t_activation(self.t_linear1(text_src2))))
        img_src = img_src + self.v_dropout2(img_src2)
        text_src = text_src + self.t_dropout2(text_src2)

        return text_src, img_src

    def forward(self, text_src, img_src,
                img_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(text_src, img_src, img_mask, src_key_padding_mask, pos)
        return self.forward_post(text_src, img_src, img_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    '''
    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    '''
    def forward_pre(self, t_output, v_output,
                img_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        t_output2 = self.norm1(t_output)
        #q = k = self.with_pos_embed(tgt2, query_pos)
        
        self_atten_output, self_atten_weight = self.self_attn(t_output2, t_output2, t_output2)
        t_output = t_output + self.dropout1(self_atten_output)
        
        Qt = self.norm2(t_output)
        #Vv = Kv = self.with_pos_embed(v_output, pos)
        #print(">>> G_Transformer decoder_layer - Qt Vv Kv: ", Qt.shape, Vv.shape, Kv.shape)
        #print(">>> G_Transformer decoder_layer - src_key_padding_mask: ", src_key_padding_mask.shape, src_key_padding_mask)
        multi_atten_output, multi_atten_weight = self.multihead_attn(query=Qt,
                                   key=self.with_pos_embed(v_output, pos),
                                   value=v_output) 
                                   #attn_mask=img_mask,
                                   #key_padding_mask=src_key_padding_mask)
        t_output = t_output + self.dropout2(multi_atten_output)
        
        t_output2 = self.norm3(t_output)
        t_output2 = self.linear2(self.dropout(self.activation(self.linear1(t_output2))))
        t_output = t_output + self.dropout3(t_output2)
        return t_output

    def forward(self, t_output, v_output,
                img_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(t_output, v_output, img_mask, src_key_padding_mask, pos)
        return self.forward_post(t_output, v_output, img_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_grounding_transformer(args):
    return G_Transformer(
        d_model=args.hidden_dim, #256
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward, #2048
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=False,
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
