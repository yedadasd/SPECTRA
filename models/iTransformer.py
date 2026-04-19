import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.quantiles = configs.quantiles
        self.num_quantiles = len(self.quantiles)
        self.mid_idx = self.num_quantiles // 2
        # Embedding
        # core
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.projection = nn.Linear(configs.d_model, self.num_quantiles * configs.pred_len, bias=True)

        self.get_parameter_number()
    def get_parameter_number(self):
        """
        Number of model parameters (without stable diffusion)
        """
        total_num = sum(p.numel() for p in self.parameters())
        param_memory = total_num * 4 / 1024 / 1024
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_ratio = trainable_num / total_num

        print('total_num:', total_num)
        print('param_memory:', param_memory)
        print('trainable_num:', total_num)
        print('trainable_ratio:', trainable_ratio)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # [B, L, N]
        B, _, N = x_enc.shape

        # Embedding
        # [B, N+4, D]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # [B, N, Q*L]
        dec_out = self.projection(enc_out)[:, :N, :]
        dec_out = dec_out.reshape(B, N, self.num_quantiles, self.pred_len)
        dec_out = dec_out.permute(0, 2, 3, 1)  # [B, Q, L, V]
        
        lower = -torch.abs(dec_out[:, :self.mid_idx, :, :])
        mid = dec_out[:, self.mid_idx:self.mid_idx+1, :, :]
        upper = torch.abs(dec_out[:, self.mid_idx+1:, :, :])
        dec_out = torch.cat([lower, mid, upper], dim=1)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).unsqueeze(1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).unsqueeze(1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out

        return None
