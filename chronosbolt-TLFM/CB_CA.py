import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.linalg import solve
from torch.utils.data import DataLoader, Subset
from sklearn.covariance import LedoitWolf
import random
from torch.distributions import Normal

from data import ParquetDataset
from Beta import Beta

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.best_state = None

    def step(self, current_loss, model):
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.num_bad_epochs = 0
            # save best weights
            self.best_state = {k: v.clone() for k,v in model.state_dict().items()}
            return False  # not stopping
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

class CB_CA(nn.Module):
    def __init__(self,
                 df_full, dfs, X2_df,
                 features_json: str,
                 beta_net: Beta,       # <-- Accept an encoder instance here
                 num_factors: int = 5,
                 seq_len: int = 60,
                 batch_size: int = 256,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 lambda_quantiles = 1e-1,
                 lambda_autoencoder = 1e-5,
                 tau = [0.1,0.9],
                 beta_huber = 0.5,
                 min_delta = 1e-4,
                 lambda_factors = None,
                 factor_months_reg = 64,
                 device: str = 'cuda'):
        
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.lambda_quantiles = lambda_quantiles
        self.lambda_autoencoder = lambda_autoencoder
        self.seq_len = seq_len
        self.tau_low = tau[0]
        self.tau_high = tau[1]
        self.beta_huber = beta_huber
        self.min_delta = min_delta
        self.lambda_factors = lambda_factors
        self.factor_months_reg=factor_months_reg

        # ─── Datasets & loaders ──────────────────────────────────────────────
        self.data_ds = ParquetDataset(
            dataframe = df_full,
            features_json=features_json,
            seq_len = self.seq_len,
            mode='window'
        )

        self.train_loader = self.data_ds.get_loader(
            batch_size=self.batch_size,
            shuffle=True
        )

        self.month_ds = ParquetDataset(
            dataframe = df_full,
            features_json=features_json,
            seq_len = self.seq_len,
            mode='month'
        )

        self.month_loader = self.month_ds.get_loader()
        self.Dates = self.month_ds.months
        self.num_charac  = self.data_ds.num_charac
        self.num_factors = num_factors

        # ─── Cross‐sectional RZ data & X2_t ────────────────────────
        self.RZ = pd.concat(dfs)

        X2_df = X2_df.sort_index()
                     
        # turn the Timestamp index into 'YYYY-MM-DD' strings
        X2_df.index = X2_df.index.strftime('%Y-%m-%d')
        self.X2 = {
            month: torch.from_numpy(X2_df.loc[month].values.astype(np.float32)).to(self.device)
            for month in X2_df.index
        }


        # ─── Model components ────────────────────────────────────────────────

        # Use the externally‐passed “beta_net”
        self.beta_net = beta_net.to(self.device)

        self.factors = nn.Linear(self.num_charac, self.num_factors, bias = False).to(self.device)
        nn.init.xavier_uniform_(self.factors.weight) 

        self.optim = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )



    def forward(self, X1: torch.Tensor, dates: list[str]) -> torch.Tensor:
        beta_out = self.beta_net(X1)
        x2_batch = torch.stack([self.X2[m] for m in dates], dim=0)  # [B, num_charac]

        f_out    = self.factors(x2_batch)            # [B, K]
        z        = beta_out * f_out                  # [B, K]

        med  = z.sum(dim=1)                          # median forecast
        return med


    
    def regularizer_on_batch(self, batch_dates, M, lambda_ortho, lambda_pos):
        # 1) ensure at least M months
        if len(batch_dates) < M:
            candidates = [d for d in self.Dates if d not in batch_dates]
            # if not enough left, we'll just sample all remaining
            k = min(M - len(batch_dates), len(candidates))
            batch_dates = batch_dates + random.sample(candidates, k)

        # 2) build factor matrix for those months
        F_hist = torch.stack([
            self.factors(self.X2[dt].unsqueeze(0)).squeeze(0)
            for dt in batch_dates
        ], dim=0)  # [M_or_more, K]

        # 3) orthogonality penalty (off-diagonal only)
        G = F_hist.T @ F_hist / F_hist.size(0)
        off = G - torch.diag(torch.diag(G))
        loss_ortho = lambda_ortho * torch.norm(off, p='fro')**2/ (self.num_factors*(self.num_factors-1))

        # 4) positive-mean penalty
        mean_f = F_hist.mean(dim=0)
        loss_pos = lambda_pos * torch.sum(torch.relu(-mean_f))/(self.num_factors)

        return loss_ortho + loss_pos



    def fit(self,
            epochs: int = 50,
            early_stopping: EarlyStopping = None,
            train_end: str = '2014-12-31',
            val_start: str = '2015-01-31',
            val_end: str = '2019-12-31'):

        train_idx = [i for i, m in enumerate(self.Dates) if m <= train_end]
        val_idx = [i for i, m in enumerate(self.Dates) if val_start <= m <= val_end]

        train_loader = DataLoader(
            Subset(self.month_ds, train_idx),
            batch_size=1,
            shuffle=False
        )

        val_loader = DataLoader(
            Subset(self.month_ds, val_idx),
            batch_size=1,
            shuffle=False
        )

        best_epoch = 0

        for ep in range(1, epochs + 1):
            # training pass
            self.train()
            train_loss = 0.0
            total_windows = 0
            for X1_m, Y_m, month in train_loader:
                X1_m = X1_m.squeeze(0).to(self.device)
                Y_m = Y_m.squeeze(0).to(self.device)
                dates = [month[0]] * X1_m.size(0)
                med = self.forward(X1_m,dates)

                loss  = nn.SmoothL1Loss(beta=self.beta_huber)(med,  Y_m)

                if self.lambda_factors is not None:
                    # build unique batch-dates
                    batch_dates = list(set(dates))  # e.g. ['2020-03-31', '2020-04-30', ...]

                    # add our factor-based regularizer
                    loss_reg = self.regularizer_on_batch(
                        batch_dates=batch_dates,
                        M=self.factor_months_reg,               # e.g. require at least 64 distinct months
                        lambda_ortho=self.lambda_factors,
                        lambda_pos=self.lambda_factors
                    )

                    loss = loss + loss_reg

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_loss += loss.item() * Y_m.numel()
                total_windows += Y_m.numel()
            train_loss = train_loss / total_windows

            # validation
            if early_stopping is not None:
                self.eval()
                val_loss = 0.0
                total_val = 0
                with torch.no_grad():
                    for X1_v, Y_v, month in val_loader:
                        X1_v = X1_v.squeeze(0).to(self.device)
                        Y_v = Y_v.squeeze(0).to(self.device)
                        dates = [month[0]] * X1_v.size(0)
                        med= self.forward(
                            X1_v,
                            dates
                        )
                        val_loss += nn.MSELoss()(med, Y_v).item() * Y_v.numel()
                        total_val += Y_v.numel()

                    val_loss = val_loss / total_val

                stop = early_stopping.step(val_loss, self)
                if stop:
                    break
                best_epoch = ep

            #print(f"[Epoch {ep:02d}] Train MSE={train_loss:.5f}  Val MSE={val_loss:.5f}", flush=True)

        if early_stopping is not None:
            early_stopping.restore(self)



    def fit_OOS(self,
                OOS_window: str = 'recursive',
                OOS_window_specs: int = 180,
                train_freq: int = 12,
                patience: int = 3,
                val_window_size: int = 12,
                test_window_size: int = 12,
                retrain_epochs: int = 50):
        Dates = self.Dates
        idx0  = OOS_window_specs
        total_sse = pred_sse = total_sst = 0.0
        f_hats = {}                    # Will hold torch tensors of shape [num_factors]
        r_p = {}   # month-string → realized tangency return
        r_ls_eq = {}                    # Equal-weighted factors
        r_shrunk = {}
        train_dates = Dates[:idx0]

        # Warn if validation window is smaller than the frequency of retraining
        if val_window_size < train_freq:
            print("Warning: val_window_size < train_freq; only the first train_freq months of validation window will be added to the training set.", flush=True)


        while idx0 + val_window_size + test_window_size <= len(Dates):
            # 1) Define the next validation/test windows
            val_dates  = Dates[idx0 : idx0 + val_window_size]
            test_dates = Dates[idx0 + val_window_size : idx0 + val_window_size + test_window_size]

            print(f"\nRetraining through {train_dates[-1]}, validating on {val_dates[0]}–{val_dates[-1]}", flush=True)

            # 2) Retrain up to train_dates[-1], validate on val_dates
            es = EarlyStopping(patience=patience, min_delta=self.min_delta)

            self.fit(
                epochs       = retrain_epochs,
                early_stopping = es,
                train_end    = train_dates[-1],
                val_start    = val_dates[0],
                val_end      = val_dates[-1]
            )

            # 3) Compute & store factors for every train_date and val_date
            #    (if not already in f_hats)
            for m in train_dates:
                if m not in f_hats:
                    f_hats[m] = self.factors(self.X2[m]).detach().cpu()

            for m in val_dates:
                if m not in f_hats:
                    f_hats[m] = self.factors(self.X2[m]).detach().cpu()

            # 4) Evaluate on test window, **including earlier test months in λₘ**

            hist_f = torch.stack([f_hats[d] for d in train_dates],
                      dim=0).to(self.device)
            w, w_shrunk, mu_F = self.compute_tangency_weights(hist_f, target_vol=(12 ** 0.5) * 0.1)
            w_eq = torch.ones_like(w) / w.numel()  # Equal-weighted

            self.eval()
            with torch.no_grad():
                for m in sorted(test_dates):
                    for X1_m, Y_m, month in self.month_loader:
                        if month[0] != m:
                            continue
                        X1m, Ym = X1_m.squeeze(0).to(self.device), Y_m.squeeze(0).to(self.device)

                        Bm = self.beta_net(X1m)
                        fm = self.factors(self.X2[m])  # [K]

                        # Tangency and Equal-weighted factors
                        r_p[m] = (w @ fm).item()
                        r_shrunk[m] = (w_shrunk @ fm).item()

                        # Forecasted return from model (used for decile sorting)
                        pred_ret = (Bm @ mu_F.unsqueeze(1)).squeeze(1)  # [N]
                        N = pred_ret.shape[0]
                        decile_size = max(N // 10, 1)

                        # Sort by predicted return
                        sorted_idx = pred_ret.argsort()
                        top_decile = sorted_idx[-decile_size:]
                        bottom_decile = sorted_idx[:decile_size]

                        # Long-short equal-weighted portfolio using REALIZED RETURNS
                        r_ls_eq[m] = Ym[top_decile].mean() - Ym[bottom_decile].mean()

                        # Errors
                        Yhat_pred = (Bm @ mu_F.unsqueeze(1)).squeeze(1)

                        total_sse += ((Ym - (Bm @ fm.unsqueeze(1)).squeeze(1)) ** 2).sum().item()
                        pred_sse += ((Ym - Yhat_pred) ** 2).sum().item()
                        total_sst += (Ym ** 2).sum().item()

                        f_hats[m] = fm.detach().cpu()


                    # Advance window
            train_dates += val_dates[:train_freq]
            if OOS_window == 'rolling':
                train_dates = train_dates[-OOS_window_specs:]
                f_hats = {d: f_hats[d] for d in train_dates}
            idx0 += train_freq

        # Final Metrics
        R2_tot  = 1 - total_sse / total_sst
        R2_pred = 1 - pred_sse / total_sst

        def sharpe(series):
            x = torch.tensor([series[m] for m in sorted(series)], dtype=torch.float32)
            return (x.mean() / x.std(unbiased=False)) * (12 ** 0.5)

        return {
            'R2_Total_OOS': R2_tot,
            'R2_Pred_OOS':  R2_pred,
            'Sharpe_Tangency': sharpe(r_p),
            'Sharpe_Tangency_Shrunk' : sharpe(r_shrunk),
            'Sharpe_LongShort_Equal': sharpe(r_ls_eq),
        }


    def compute_tangency_weights(self,
                                 factor_ts: torch.Tensor,
                                 target_vol: float | None = None
                                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute tangency portfolio weights using both the empirical covariance and Ledoit-Wolf shrinkage.

        Returns:
            w_emp: Tangency weights from empirical covariance.
            w_lw:  Tangency weights from Ledoit-Wolf covariance.
            mu_F:  Sample mean of factor returns.
        """
        # factor_ts: Tensor[T, K]
        T, K = factor_ts.shape

        # 1) Sample mean
        mu_F = factor_ts.mean(dim=0)  # [K]

        # 2a) Empirical covariance estimate
        X = factor_ts - mu_F                   # Centered returns [T, K]
        Sigma_emp = (X.T @ X) / (T - 1)        # Unbiased sample covariance [K, K]

        # 2b) Ledoit-Wolf covariance estimate
        lw = LedoitWolf().fit(factor_ts.cpu().numpy())
        Sigma_lw = torch.from_numpy(lw.covariance_).to(factor_ts.device)  # [K, K]

        # 3a) Raw tangency weights (empirical)
        w_emp = torch.linalg.solve(Sigma_emp, mu_F)
        # 3b) Raw tangency weights (Ledoit-Wolf)
        w_lw = torch.linalg.solve(Sigma_lw, mu_F)

        # 4) Scale to target vol or sum to 1
        def scale_weights(w: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
            if target_vol is not None:
                sigma_p = torch.sqrt((w @ Sigma) @ w).clamp(min=1e-12)
                return w * (target_vol / sigma_p)
            else:
                return w / w.sum()

        w_emp = scale_weights(w_emp, Sigma_emp)
        w_lw = scale_weights(w_lw, Sigma_lw)

        return w_emp, w_lw, mu_F
