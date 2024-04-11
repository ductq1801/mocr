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
class Attention(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs

class Image_Encoder_v2(nn.Module):
    def __init__(self,hidden_dim,pretrained=True,drop_out=0.2):
        super(Image_Encoder,self).__init__()
        model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k',pretrained=pretrained)
        self.trans = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.features = nn.Sequential(*list(model.children())[:-4])
        in_features = self.features[-1][-1][-1].conv_pwl.out_channels
        self.dropout = nn.Dropout(drop_out)
        self.conv_1x1 = nn.Conv2d(in_features, hidden_dim, 1)
        self.attn = Attention(hidden_dim)
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
        x = x + self.attn(x)
        x = x.permute(1, 0, -1)
        return x