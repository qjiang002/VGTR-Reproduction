"""
Backbone modules.
"""
import torch
import torch.nn.functional as F

from torch import embedding, nn
from typing import Dict, List

from utils.misc import NestedTensor

class BiLSTM_Att(nn.Module):
    def __init__(self, train_bilstm: bool, hidden_dim: int, max_len: int, embedding_dim: int, num_layers: int, num_channels: int, dropout: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.bi_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.atten = nn.Linear(2*hidden_dim, num_channels, bias=False)
        
        if not train_bilstm:
            for parameter in self.bi_lstm.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        input_emb = self.embedding(tensor_list.tensors) #batch, seq_len, embedding_size
        output, (_, _) = self.bi_lstm(input_emb)
        #print("BiLSTM_Att output: ", output.shape)
        att_weights = F.softmax(self.atten(output), 1).permute(0, 2, 1) #batch, num_channel, seq_len
        text_out = torch.matmul(att_weights, input_emb) #batch, num_channel, embedding_size
        #print("BiLSTM_Att text_out: ", text_out.shape, self.num_channels, self.embedding_dim)
        #assert text_out.shape[1:] == [self.num_channels, self.embedding_dim]
        #print("BiLSTM_Att text_out: ",text_out.shape)
        
        out = NestedTensor(text_out, None)

        # Output [Batch, TimeStep, EmbeddingDim] vec + reversed mask
        return out


    def assign_embedding(self, embedding_matrix, num_embeddings: int, embedding_dim: int):
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if embedding_matrix == None:
            return

        self.embedding.weight = embedding_matrix


def build_vgtr_language(args):
    train_bilstm = args.lr_bilstm > 0
    lstm = BiLSTM_Att(train_bilstm, args.bilstm_hidden_dim, args.max_query_len, args.embedding_dim, args.bilstm_layers, args.bilstm_out_dim, args.bilstm_dropout)
    return lstm
