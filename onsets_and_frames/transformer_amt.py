import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
class Transformer_AMT(nn.Module):
    """
    This is adapted from the nn.Transformer with the following modification:
      (a). No decoder
      (b). input  and output shape is (batch_size, sequence_len, num_features)
        (that is batch_first=True as in biLSTM) 
      (c). added a Linear and sigmoid 
    """
    def __init__(self, d_model,
                 nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):

        assert d_model % nhead == 0 
        super().__init__()
        #sgu: hidden_size = recurrent_features
        # since batch_first = True, input and output shape is (batch, seq, feature)
        encoder_layer = TransformerEncoderLayer(d_model, nhead,
                                    dim_feedforward, dropout, activation)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                         encoder_norm)
        
    def forward(self, x):
        """
        :param x: nn.tensor with shape (batch_size, seq_len, num_features)
        :return : nn.tensor with same shape of x
        """
        x = x.transpose(0,1) 
        rv = self.encoder(x).transpose(0,1) 
        return rv 
