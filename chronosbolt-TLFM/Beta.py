

import torch
import torch.nn as nn
import torch.nn.functional as F

class Beta(nn.Module):
    def __init__(
        self,
        chronos_pipeline,
        num_characs: int = 94,
        num_factors: int = 5,
        num_heads: int = 8,
        head_dim:  int = 32,
        use_hidden_states: bool = False,
        token_mode: str = "reg_and_last",
        pca_dim: int = 10
    ):
        super().__init__()
        # ── Underlying frozen Bolt model ─────────────────────────
        self.bolt_model = chronos_pipeline.model
        for p in self.bolt_model.parameters():
            p.requires_grad = False
        # unfreeze final block
        #for p in self.bolt_model.encoder.block[-1].parameters():
        #    p.requires_grad = True

        self.num_characs      = num_characs
        self.pca_dim          = pca_dim
        self.use_hidden_states= use_hidden_states
        self.token_mode       = token_mode
        self.d_model = self.bolt_model.config.d_model

        # ── 1) Non-linear autoencoder bottleneck ─────────────────
        if pca_dim is not None:
            hidden = (num_characs + pca_dim) // 2
            self.encoder_pca = nn.Sequential(
                nn.Linear(num_characs, hidden),
                #nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, pca_dim),
                #nn.BatchNorm1d(pca_dim)
            )

                        # decoder: mirror of encoder (no activation on last layer)
            self.decoder_pca = nn.Sequential(
                nn.Linear(pca_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, num_characs),
            )

                        # init weights
            for m in list(self.encoder_pca) + list(self.decoder_pca):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            #    elif isinstance(m, nn.BatchNorm1d):
            #        # scale=1, shift=0
            #       nn.init.constant_(m.weight, 1.0)
             #       nn.init.constant_(m.bias,   0.0)

        else:
            self.encoder_pca = None




        # ── 2) Query token & cross-attention ─────────────────────
        

        # 1) Decide your reduced dimension
        self.num_heads       = num_heads
        self.head_dim        = head_dim
        self.reduced_d_model = num_heads * head_dim
        assert self.reduced_d_model <= self.d_model, \
            f"reduced_d_model={self.reduced_d_model} must be ≤ {self.d_model}"
        
        self.query_token = nn.Parameter(torch.randn(1, 1, self.reduced_d_model))

        # 2) A small linear to project down
        self.attn_in_proj = nn.Linear(self.d_model, self.reduced_d_model)

        # 3) Multi-head attention on the reduced space
        self.attn1 = nn.MultiheadAttention(
            embed_dim=self.reduced_d_model,
            num_heads=self.num_heads,
            batch_first=True
        )


        # ── 3) Final factor projection ───────────────────────────
        self.final_linear = nn.Linear(self.reduced_d_model, num_factors, bias=True)
        nn.init.xavier_uniform_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            nn.init.zeros_(self.final_linear.bias)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
    
        #X: (B, seq_length, num_characs)
        #Returns: (B, num_factors)
        
        B, seq_len, C0 = X.shape

        # —— Autoencoder bottleneck ——  


        if self.encoder_pca is not None:
            X0 = X.view(-1, C0)              # (B*seq_len, num_characs)
            Z  = self.encoder_pca(X0)       # (B*seq_len, pca_dim)
            X_rec = self.decoder_pca(Z)     # reconstruct back to num_characs

            ae_loss = F.mse_loss(X_rec, X0)  
            X = Z.view(B, seq_len, -1)      # now (B, seq_len, pca_dim)
            ts_dim = self.pca_dim
        else:
            ae_loss = X.new_tensor(0.)
            ts_dim = self.num_characs
    

        # —— Bolt encoding ——  
        # flatten each series for encoding
        # we permute so that each "characteristic" is treated as its own sequence
        X_flat = X.permute(0,2,1).reshape(B * ts_dim, seq_len)
        with torch.no_grad():
            hidden_flat, *_ = self.bolt_model.encode(X_flat)

        # build encoder_outputs of shape (B, ts_dim * L, d_model)
        if self.use_hidden_states:
            # hidden_flat: (B*ts_dim, seq_hidden, d_model)
            hs = hidden_flat.view(B, ts_dim, -1, self.d_model)
            encoder_outputs = hs.view(B, ts_dim * hs.size(2), self.d_model)
        else:
            if self.token_mode == "reg_and_last":
                last_two = hidden_flat[:, -2:, :]               # (B*ts_dim, 2, d_model)
                encoder_outputs = last_two.reshape(B, ts_dim*2, self.d_model)
            elif self.token_mode == "last_patch":
                lp = hidden_flat[:, -2, :]                     # (B*ts_dim, d_model)
                encoder_outputs = lp.view(B, ts_dim, self.d_model)
            else:  # "reg"
                reg = hidden_flat[:, -1, :]
                encoder_outputs = reg.view(B, ts_dim, self.d_model)

        # —— Cross‐attention pooling ——  
                # project down:
        proj = self.attn_in_proj(encoder_outputs)  # (B, L, reduced_d_model)

        # attention:
        query = self.query_token.expand(B, -1, -1)  # (B,1,reduced_d_model)
        a1, _ = self.attn1(query, proj, proj)       # (B,1,reduced_d_model)

        agg = a1.squeeze(1)                # (B,d_model)

        # —— Project to factors ——  
        return self.final_linear(agg)
