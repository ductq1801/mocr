import timm
import torch
from torch import nn

class Image_Encoder(nn.Module):
    def __init__(self,hidden_dim,pretrained=True,drop_out=0.2):
        super(Image_Encoder,self).__init__()
        self.model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k',pretrained=pretrained)
        in_feature = self.model.classifier.infeatures
        self.model.classifier = nn.Sequential(nn.Dropout(drop_out),
                                              nn.Linear(in_feature,hidden_dim))
    def forward(self,x):
        """
        Shape:
            - x: (B, C, W, H)
            - output: (B, H_dim)
            
        """
        return self.model(x)
