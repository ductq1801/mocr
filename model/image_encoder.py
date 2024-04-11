import timm
import torch
from torch import nn
import torch.nn.functional as F
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from math import sqrt

class Image_Encoder(nn.Module):
    def __init__(self,hidden_dim,pretrained=True,drop_out=0.2):
        super(Image_Encoder,self).__init__()
        model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k',pretrained=pretrained)
        self.trans = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.features = nn.Sequential(*list(model.children())[:-4])
        in_features = self.features[-1][-1][-1].conv_pwl.out_channels
        self.dropout = nn.Dropout(drop_out)
        self.conv_1x1 = nn.Conv2d(in_features, hidden_dim, 1)
    def forward(self,x):
        """
        Shape:
            - x: (B, C, W, H)
            - output: (H_dim,B, C)
            
        """
        x = self.features(x)
        x = self.dropout(x)
        x = self.conv_1x1(x)
        x =  x.transpose(-1, -2)
        x = x.flatten(2)
        x = x.permute(-1, 0, 1)
        return x


def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)
class AttentionHead(nn.Module):
    def __init__(self, embed_dim=512, head_dim=None):
        super().__init__()
        if head_dim == None:
            head_dim = embed_dim
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        embed_dim = embed_dim # 512
        num_heads = embed_dim // 4 # 128
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
class Image_Encoder_v2(nn.Module):
    def __init__(self,hidden_dim,pretrained=True,drop_out=0.2):
        super(Image_Encoder,self).__init__()
        model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k',pretrained=pretrained)
        self.trans = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.features = nn.Sequential(*list(model.children())[:-4])
        in_features = self.features[-1][-1][-1].conv_pwl.out_channels
        self.dropout = nn.Dropout(drop_out)
        self.conv_1x1 = nn.Conv2d(in_features, hidden_dim, 1)
        self.attn = MultiHeadAttention(hidden_dim)
        self.norm = LayerNormalization(hidden_dim)
    def forward(self,x):
        """
        Shape:
            - x: (B, C, W, H)
            - output: (H_dim,B, C)
            
        """
        x = self.features(x)
        x = self.dropout(x)
        x = self.conv_1x1(x)
        x = x.transpose(-1, -2)
        x = x.flatten(2)
        x = x.permute(0, -1, 1)
        x = x + self.attn(self.norm(x))
        x = x.permute(1, 0, -1)
        return x