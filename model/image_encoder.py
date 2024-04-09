import timm
import torch
from torch import nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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
