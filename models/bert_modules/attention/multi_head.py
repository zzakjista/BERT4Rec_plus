import torch
import torch.nn as nn
from .single import Attention

item_eigen = torch.load('item_eigen.pt')
user_eigen = torch.load('user_eigen.pt')
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        item_eigen = torch.load('item_eigen.pt')
        user_eigen = torch.load('user_eigen.pt')

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        
        self.query_weight = user_eigen
        self.key_weight = item_eigen
        self.value_weight = torch.matmul(user_eigen.transpose(0,1), item_eigen)
        
        with torch.no_grad():
            self.linear_layers[0].weight.copy_(self.query_weight)
            self.linear_layers[1].weight.copy_(self.key_weight)
            self.linear_layers[2].weight.copy_(self.value_weight)
        

        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
