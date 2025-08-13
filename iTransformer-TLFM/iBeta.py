import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class iBeta(nn.Module):
    """

    Input:  
        x_enc:      [B, seq_len=60, N=94]
    Output:
        factors: [B, num_factors=5]
    """

    def __init__(self, configs, num_heads: int = 8):
        super(iBeta, self).__init__()

        self.seq_len     = configs.seq_len       # 60
        self.num_tokens  = configs.N             # 94
        self.num_factors = configs.num_factors     # 5

        # 1) Inverted embedding: each of the 94 variables → a token of size d_model.
        #    Now we will pass x_mark_enc into it (instead of always None).
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # 2) Inverted Transformer (stacked encoder layers)
        encoder_layers = []
        for _ in range(configs.e_layers):
            encoder_layers.append(
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
            )
        self.encoder = Encoder(
            encoder_layers,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # 3) Instead of a per-token projector, we pool the 94 tokens → [B, d_model],
        #    then project to num_factors=5:
        self.pool             = nn.AdaptiveAvgPool1d(1)
        self.final_linear = nn.Linear(configs.d_model, configs.num_factors, bias=True)

        self.use_norm         = configs.use_norm
        self.output_attention = configs.output_attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, configs.d_model))

        
    def forward(self, x_enc, x_mark_enc=None):
        """
        x_enc:      [B, seq_len=60, N=94]
        x_mark_enc: [B, seq_len=60, d_time]  (or None, if you have no date covariates)
        Returns:
            factors: [B, 5]
            (attns if output_attention=True)
        """
        B,_,_ = x_enc.shape
        # 1) (Optional) normalize each series over time
        if self.use_norm:
            # x_enc: [B, 60, 94] → compute mean & stdev over dim=1 (time)
            means = x_enc.mean(dim=1, keepdim=True).detach()             # [B, 1, 94]
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # 2) Inverted embedding:
        #    Now pass x_mark_enc instead of None.
        #    If x_mark_enc is None, DataEmbedding_inverted should handle it internally.
        #    Input:  x_enc [B, 60, 94], x_mark_enc [B, 60, d_time] or None
        #    Output: enc_out [B, 94, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  

        cls_tokens = self.cls_token.expand(B, -1, -1)     # [B, 1, d_model]
        enc_in = torch.cat([enc_out, cls_tokens], dim=1)  # [B, C+1, d_model]
        enc_out_with_cls, attns = self.encoder(enc_in, attn_mask=None)
        # After encoding:
        cls_embedding = enc_out_with_cls[:, -1, :]           # [B, d_model]
        factors = self.final_linear(cls_embedding)          # [B, num_factors]
        return factors  # if you want the attention weights


# 1) Define the same Configs class as before. No changes needed here.
class Configs:
    """
    Configuration container for the iBeta / iCA models.

    Args:
        seq_len (int): length of each input time series (default: 45)
        N (int): number of variables/tokens (default: 94)
        num_factors (int): number of common factors to output (default: 5)
        d_model (int): transformer embedding dimension
        d_ff (int): feedforward network dimension
    """
    def __init__(self,
                 seq_len: int = 45,
                 N: int = 94,
                 num_factors: int = 5,
                 d_model: int = 128,
                 d_ff: int = 256,
                 embed = None,
                 freq = None,
                 n_heads: int = 4,
                 e_layers: int = 2,
                 factor: int = 5,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 use_norm: bool = False,
                 output_attention: bool = False,
                 class_strategy = None
                 ):
        self.seq_len = seq_len      # length of each input series
        self.N = N                  # number of variables (tokens)
        self.num_factors = num_factors  # number of common factors to output

        # Embedding / transformer dimensions:
        self.d_model = d_model
        self.embed = embed          # time‐feature embedding (None to disable)
        self.freq = freq            # time‐feature frequency (None to disable)
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.use_norm = use_norm
        self.output_attention = output_attention
        self.class_strategy = class_strategy

