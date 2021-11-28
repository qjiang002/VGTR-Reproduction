"""
Backbone modules.
"""
import torch
import torch.nn.functional as F

from torch import embedding, nn
from typing import Dict, List

from utils.misc import NestedTensor

class BiLSTM(nn.Module):
    def __init__(self, name: str, train_bilstm: bool, hidden_dim: int, max_len: int, embedding_dim: int, num_layers: int, num_channels: int, dropout: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.bi_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.proj_layer = torch.nn.Linear(hidden_dim, num_channels)

        if not train_bilstm:
            for parameter in self.bi_lstm.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        input_emb = self.embedding(tensor_list.tensors)

        output, (_, _) = self.bi_lstm(input_emb)

        output = torch.mean(output.view(output.shape[0], output.shape[1], 2, self.hidden_dim), dim=2) # Take the mean of both direction
        output = self.proj_layer(output)  # project so that it has num_channels channels

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(output, mask)

        # Output [Batch, TimeStep, EmbeddingDim] vec + reversed mask
        return out


    def assign_embedding(self, embedding_matrix, num_embeddings: int, embedding_dim: int):
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if embedding_matrix == None:
            return

        self.embedding.weight = embedding_matrix


def build_bilstm(args):
    # position_embedding = build_position_encoding(args)
    train_bilstm = args.lr_bilstm > 0
    lstm = BiLSTM(args.bert_model, train_bilstm, args.bilstm_hidden_dim, args.max_query_len, args.embedding_dim, args.bilstm_layers, args.bilstm_out_dim, args.bilstm_dropout)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels
    return lstm
