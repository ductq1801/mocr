import timm
import torch
from torch import nn
import torch.nn.functional as F
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


class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C) # global average pooling
        return a, output

class Image_Encoder_v2(nn.Module):
    def __init__(self,hidden_dim,pretrained=True,drop_out=0.2,normalize_attn=True):
        super(Image_Encoder,self).__init__()
        model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k',pretrained=pretrained)
        self.trans = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        self.features = nn.Sequential(*list(model.children())[:-4])
        in_features = self.features[-1][-1][-1].conv_pwl.out_channels
        self.dropout = nn.Dropout(drop_out)
        self.conv_1x1 = nn.Conv2d(in_features, hidden_dim, 1)
        self.pool = nn.AvgPool2d(2, stride=1)
        self.attn1 = AttentionBlock(hidden_dim, hidden_dim, hidden_dim, 1, normalize_attn=normalize_attn)
    def forward(self,x):
        """
        Shape:
            - x: (B, C, W, H)
            - output: (H_dim,B, C)
            
        """
        x1 = self.features(x)
        x1 = self.dropout(x1)
        x2 = self.conv_1x1(x1)
        pool1 = F.max_pool2d(x1, 2, 2)
        pool2 = F.max_pool2d(x2, 2, 2)
        a,g = self.attn1(pool1,pool2)
        pool2 = self.pool(pool2)
        pool2 =  pool2.transpose(-1, -2)
        pool2 = pool2.flatten(2)
        pool2 = pool2.permute(0, -1, 1)
        g = g.unsqueeze(1)
        x = torch.cat((pool2,g),dim=1)


        # x =  x.transpose(-1, -2)
        # x = x.flatten(2)
        #x = x.permute(-1, 0, 1)
        x = x.permute(1, 0, -1)
        return x