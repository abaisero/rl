import torch
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    def __init__(self, nnodes, gain=1.):
        super().__init__()
        embeddings = torch.eye(nnodes)
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.affine = nn.Linear(nnodes, 1, bias=True)
        nn.init.xavier_normal_(self.affine.weight, gain)
        nn.init.constant_(self.affine.bias, 0)

    def forward(self, n):
        em = self.embedding(n)
        return self.affine(em)
