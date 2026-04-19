import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True): 
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        
    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


#TiDE
class Model(nn.Module):  
    """
    paper: https://arxiv.org/pdf/2304.08424.pdf 
    """
    def __init__(self, configs, bias=True, feature_encode_dim=2): 
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  #L 
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len  #H 
        self.quantiles = configs.quantiles
        self.num_quantiles = len(self.quantiles)
        self.mid_idx = self.num_quantiles // 2
        self.hidden_dim=configs.d_model
        self.res_hidden=configs.d_model 
        self.encoder_num=configs.e_layers
        self.decoder_num=configs.d_layers
        self.freq=configs.freq
        self.feature_encode_dim=feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporalDecoderHidden=configs.d_ff
        dropout=configs.dropout

        
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        
        self.feature_dim=freq_map[self.freq]


        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias)
        self.encoders = nn.Sequential(ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias),*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.encoder_num-1)))
        
        self.decoders = nn.Sequential(*([ ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias)]*(self.decoder_num-1)),ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len * self.num_quantiles, dropout, bias))
        self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias)
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)
            
        
    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        feature = self.feature_encoder(batch_y_mark)
        hidden = self.encoders(torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1))
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.pred_len, self.decode_dim, self.num_quantiles)
        decoded = decoded.permute(0, 3, 1, 2)

        feature_future = feature[:, self.seq_len:, :].unsqueeze(1).repeat(1, self.num_quantiles, 1, 1)
        temporal_in = torch.cat([feature_future, decoded], dim=-1)
        bsz, q_num, out_len, feat_dim = temporal_in.shape
        temporal_in = temporal_in.reshape(bsz * q_num, out_len, feat_dim)
        dec_out = self.temporalDecoder(temporal_in).squeeze(-1)
        dec_out = dec_out.reshape(bsz, q_num, out_len)

        residual = self.residual_proj(x_enc).unsqueeze(1)
        dec_out = dec_out + residual

        lower = -torch.abs(dec_out[:, :self.mid_idx, :])
        mid = dec_out[:, self.mid_idx:self.mid_idx + 1, :]
        upper = torch.abs(dec_out[:, self.mid_idx + 1:, :])
        dec_out = torch.cat([lower, mid, upper], dim=1)
        
        
        # De-Normalization 
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).unsqueeze(1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).unsqueeze(1))
        return dec_out
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask=None):
        '''x_mark_enc is the exogenous dynamic feature described in the original paper'''
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if batch_y_mark is None:
                batch_y_mark = torch.zeros((x_enc.shape[0], self.seq_len+self.pred_len, self.feature_dim)).to(x_enc.device).detach()
            else:
                batch_y_mark = torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]],dim=1)
            dec_out = torch.stack([self.forecast(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark) for feature in range(x_enc.shape[-1])],dim=-1)
            return dec_out # [B, Q, L, D]
        return None
    
    



