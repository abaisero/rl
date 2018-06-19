import torch
import torch.nn as nn


class MultiEmbedding_from_pretrained(nn.Module):
    # TODO avoid device init
    def __init__(self, embeddings):
        super().__init__()
        idims = torch.tensor(embeddings.shape[:-1])
        odim = embeddings.shape[-1]

        self.midx = idims.cumprod(dim=0).unsqueeze(dim=0) / idims
        embeddings_ = embeddings.reshape((-1, odim))
        self.embedding = nn.Embedding.from_pretrained(embeddings_)

    def forward(self, *codes):
        code = torch.stack(codes, dim=-1)

        # torch.bmm on CUDA does not support integer tensors
        # code = torch.bmm(
        #     self.midx.expand_as(code).unsqueeze(-1),
        #     code.unsqueeze(-1)
        # )
        code = (code * self.midx.to(code)).sum(1)
        return self.embedding(code)
