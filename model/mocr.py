from model.image_encoder import Image_Encoder
from model.transformer import Seq2Seq
from torch import nn

class MOCR(nn.Module):
    def __init__(self, vocab_size,
                 ie_args, 
                 transformer_args):
        
        super(MOCR, self).__init__()
        
        self.img_enc = Image_Encoder(**ie_args)
        self.transformer = Seq2Seq(vocab_size, **transformer_args)

    def forward(self, img, tgt_input):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - output: b t v
        """
        src = self.img_enc(img)
        outputs = self.transformer(src, tgt_input)
        return outputs