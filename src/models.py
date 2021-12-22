import torch
import torch.nn as nn
import torch.nn.functional as F


class preexp_retrieval(nn.Module):
    def __init__(self, feat_dim=0):
        super(preexp_retrieval, self).__init__()
        self.activation_func = nn.Tanh()
        self.linear_fq = nn.Linear(feat_dim, 64)
        self.linear_fk = nn.Linear(feat_dim, 64)

    def forward(self, feat_query, feat_key):
        # mapping
        q = self.linear_fq(feat_query)
        q = self.activation_func(q)
        k = self.linear_fk(feat_key)
        k = self.activation_func(k)
        k = torch.transpose(k, 0, 1)
        return F.softmax(torch.squeeze(torch.mm(q, k)), dim=0)
