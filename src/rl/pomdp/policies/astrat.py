import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class AStrategy(nn.Module):
    def __init__(self, nnodes, nactions, gain=1.):
        super().__init__()
        embeddings = torch.eye(nnodes)
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.linear = nn.Linear(nnodes, nactions, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain)

    def forward(self, n):
        em = self.embedding(n)
        scores = self.linear(em)
        return F.softmax(scores, dim=-1)

    def sample(self, n):
        probs = self(n)
        dist = Categorical(probs)
        sample = dist.sample()
        nll = -dist.log_prob(sample)
        return sample, nll
