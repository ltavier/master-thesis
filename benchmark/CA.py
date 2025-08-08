import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.linalg import solve, cho_factor, cho_solve
from torch.utils.data import DataLoader, Dataset, Subset
import json
from sklearn.covariance import LedoitWolf
import random
from torch.distributions import Normal


class EarlyStopping:
    def __init__(self, patience=5, min_delta = 1e-4):
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
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False  # not stopping
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

class RZMonthDataset(Dataset):
    """
    Dataset for full‐month observations. Groups rows of RZ by end_date.
    Returns X1 of shape (N_m, num_charac), Y of shape (N_m,), and month string.
    """
    def __init__(self, rz_files, features_json, device='cpu'):
        if isinstance(rz_files, (list, tuple)):
            dfs = [pd.read_parquet(p) for p in rz_files]
            df = pd.concat(dfs, ignore_index=False)
        else:
            df = pd.read_parquet(rz_files)

        df = df.reset_index()
        df['end_date'] = pd.to_datetime(df['end_date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values('end_date').reset_index(drop=True)
        self.device = device

        # Group row indices by month
        self.months = []
        self.indices = []
        for mon, sub in df.groupby('end_date'):
            self.months.append(mon)
            self.indices.append(sub.index.values)

        # Store raw arrays
        self.Y_arr = df['Y'].values.astype(np.float32)
        with open(features_json, 'r') as f:
            features = json.load(f)
        char_list = features
        self.num_charac = len(char_list)
        char_cols = [f"char_{i}" for i in range(self.num_charac)]
        self.X1_arr = df[char_cols].values.astype(np.float32)  # shape (N_all, num_charac)

    def __len__(self):
        return len(self.months)

    def __getitem__(self, idx):
        inds = self.indices[idx]
        raw_X1 = self.X1_arr[inds]  # shape (N_m, num_charac)
        Y = self.Y_arr[inds]        # shape (N_m,)

        X1 = torch.tensor(raw_X1, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        month = self.months[idx]
        return X1, Y, month

    def get_loader(self, num_workers=2):
        return DataLoader(
            self, batch_size=1, shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device != 'cpu')
        )


class CA(nn.Module):
    def __init__(self,
                 rz_files: list,
                 x2_parquet: str,
                 features_json: str,
                 layers: int = 1,
                 num_factors: int = 5,
                 batch_size: int = 256,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 beta_huber = 0.5,
                 lambda1 = 1e-3,
                 lambda_factors = None,
                 factor_months_reg = 64,
                 min_delta = 1e-4,
                 device: str = 'cuda'):

        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.beta_huber = beta_huber
        self.lambda1 = lambda1
        self.lambda_factors = lambda_factors
        self.factor_months_reg = factor_months_reg
        self.min_delta = min_delta

        # ─── Datasets & loaders ──────────────────────────────────────────────
        self.month_ds = RZMonthDataset(rz_files, features_json)
        self.month_loader = self.month_ds.get_loader()
        self.Dates = self.month_ds.months

        self.num_charac = self.month_ds.num_charac
        self.num_factors = num_factors

        # ─── Cross‐sectional RZ data & X2_t ────────────────────────
        X2_df = pd.read_parquet(x2_parquet)
        X2_df = X2_df.sort_index()
        X2_df.index = X2_df.index.strftime('%Y-%m-%d')
        self.X2 = {
            month: torch.from_numpy(X2_df.loc[month].values.astype(np.float32)).to(self.device)
            for month in X2_df.index
        }

        # ─── Build beta_net based on layers parameter ─────────────────────────────────
        hidden_sizes = []
        if layers >= 1:
            hidden_sizes.append(32)
        if layers >= 2:
            hidden_sizes.append(16)
        if layers >= 3:
            hidden_sizes.append(8)

        modules = []
        in_dim = self.num_charac
        for h in hidden_sizes:
            # 1) create the layer
            linear = nn.Linear(in_dim, h)
            # 2) initialize weights & biases
            nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            modules.append(linear)

            # 3) create the batch‐norm
            bn = nn.BatchNorm1d(h)
            # 4) initialize its parameters
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)
            modules.append(bn)

            # 5) activation
            modules.append(nn.ReLU())
            in_dim = h
        # Final factor‐projection layer
        final_linear = nn.Linear(in_dim, self.num_factors)
        # You may choose Xavier here to keep factors on a balanced scale:
        nn.init.xavier_uniform_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        modules.append(final_linear)

        self.beta_net = nn.Sequential(*modules).to(self.device)


        # ─── Factor loadings ───────────────────────────────────────────
        self.factors = nn.Linear(self.num_charac, self.num_factors, bias = False).to(self.device)
        nn.init.xavier_uniform_(self.factors.weight)
        #nn.init.orthogonal_(self.factors.weight)

        # ─── Optimizer ────────────────────────────────────────────────
        self.optim = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Global factor memory
        self.global_factors: dict[str, torch.Tensor] = {}

    def forward(self, X1: torch.Tensor, dates: list[str]) -> torch.Tensor:
        beta_out = self.beta_net(X1)
        x2_batch = torch.stack([self.X2[d] for d in dates], dim=0)
        f_out = self.factors(x2_batch)
        return (beta_out * f_out).sum(dim=1)

        
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
    
    def fit(
            self,
            epochs: int = 50,
            early_stopping: EarlyStopping = None,
            train_end: str = '2014-12-31',
            val_start: str = '2015-01-31',
            val_end: str = '2019-12-31'
        ):


        train_idx = [i for i, m in enumerate(self.Dates) if m <= train_end]
        train_loader = DataLoader(
            Subset(self.month_ds, train_idx),
            batch_size=1,
            shuffle=False
        )

        val_idx = [i for i, m in enumerate(self.Dates) if m >= val_start and m <= val_end]
        val_loader = DataLoader(
            Subset(self.month_ds, val_idx),
            batch_size=1,
            shuffle=False
        )


        best_epoch = 0
        for ep in range(1, epochs + 1):
            # --- Training pass (with per-batch orthogonality & positive-mean) ---
            self.train()
            train_loss = 0.0

            for X1_m, Y_m, month in train_loader:
                # X1_m: [N_m, P]; Y_m: [N_m]; month: str
                X1_m, Y_m = X1_m.to(self.device), Y_m.to(self.device)
                X1_m = X1_m.squeeze(0)  # now [Nₘ, P]
                Y_m  = Y_m.squeeze(0)   # now [Nₘ]

                # → forward cross‐sectional predictions
                beta = self.beta_net(X1_m)                            # [N_m, K]
                f_t  = self.factors(self.X2[month[0]]).unsqueeze(0)      # [1, K]
                f_t  = f_t.expand(beta.size(0), -1)                   # [N_m, K]
                preds = (beta * f_t).sum(dim=1)

                # → reconstruction loss
                base_loss = nn.SmoothL1Loss(beta=self.beta_huber)(preds, Y_m)
                l1_term = self.lambda1 * sum(p.abs().sum() for p in self.beta_net.parameters())
                loss = base_loss + l1_term

                if self.lambda_factors is not None:
                    # build unique batch-dates
                    batch_dates = list(set(month))  # e.g. ['2020-03-31', '2020-04-30', ...]

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
                train_loss += loss.item() * Y_m.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validation pass ---
            if early_stopping:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X1, Y, month in val_loader:
                    # X1_m: [N_m, P]; Y_m: [N_m]; month: str
                        X1, Y = X1.to(self.device), Y.to(self.device)
                        X1 = X1.squeeze(0)  # now [Nₘ, P]
                        Y  = Y.squeeze(0)   # now [Nₘ]

                        # → forward cross‐sectional predictions
                        beta = self.beta_net(X1)                            # [N_m, K]
                        f_t  = self.factors(self.X2[month[0]]).unsqueeze(0)      # [1, K]
                        f_t  = f_t.expand(beta.size(0), -1)                   # [N_m, K]
                        preds = (beta * f_t).sum(dim=1)
                        val_loss += ((Y - preds) ** 2).mean().item() * Y.size(0)
                val_loss /= len(val_loader.dataset)
                if early_stopping.step(val_loss, self):
                    break
                best_epoch = ep

        # Restore best weights
        if early_stopping:
            early_stopping.restore(self)

        # Apply PCA_positivemean rotation to identify factors
        #self.apply_rotation()



    def fit_OOS(self,
                OOS_window: str = 'recursive',
                OOS_window_specs: int = 180,
                train_freq: int = 12,
                patience: int = 3,
                val_window_size: int = 12,
                test_window_size: int = 12,
                retrain_epochs: int = 50):
        """
        Recursive (or rolling) OOS evaluation. Returns:
        - R2_Total_OOS
        - R2_Pred_OOS
        - Sharpe_Ratio         (tangency portfolio)
        - Sharpe_EW_Factors    (equal-weighted factor portfolio)
        - Sharpe_LS            (equal-weighted long-short portfolio)
        """

        Dates = self.Dates
        idx0 = OOS_window_specs

        total_sse = pred_sse = total_sst = 0.0
        f_hats = {}
        r_p = {}      # tangency
        r_shrunk = {}
        r_eq = {}     # equal-weighted factors
        r_ls = {}     # long-short

        train_dates = Dates[:idx0]

        while idx0 + val_window_size + test_window_size <= len(Dates):
            val_dates  = Dates[idx0 : idx0 + val_window_size]
            test_dates = Dates[idx0 + val_window_size : idx0 + val_window_size + test_window_size]

            print(f"\nRetraining through {train_dates[-1]}, validating on {val_dates[0]}–{val_dates[-1]}", flush=True)

            # Retrain
            es = EarlyStopping(patience=patience, min_delta=self.min_delta)
            self.fit(
                epochs=retrain_epochs,
                early_stopping=es,
                train_end=train_dates[-1],
                val_start=val_dates[0],
                val_end=val_dates[-1]
            )

            for m in train_dates + val_dates:
                if m not in f_hats:
                    f_hats[m] = self.factors(self.X2[m]).detach().cpu()

            hist_f = torch.stack([f_hats[d] for d in train_dates],
                                dim=0).to(self.device)
            w, w_shrunk, mu_F = self.compute_tangency_weights(hist_f, target_vol = (12 ** 0.5) * 0.1)
            w_eq = torch.ones_like(w) / w.numel()

            self.eval()
            with torch.no_grad():
                for m in sorted(test_dates):
                    for X1_m, Y_m, month in self.month_loader:
                        if month[0] != m:
                            continue

                        X1m = X1_m.squeeze(0).to(self.device)
                        Ym  = Y_m.squeeze(0).to(self.device)

                        Bm = self.beta_net(X1m)
                        fm = self.factors(self.X2[m])

                        # R2_Total
                        Yhat_in = (Bm @ fm.unsqueeze(1)).squeeze(1)
                        total_sse += ((Ym - Yhat_in) ** 2).sum().item()
                        total_sst += (Ym ** 2).sum().item()

                        f_hats[m] = fm.detach().cpu()

                        # Sharpe: Tangency
                        r_p[m] = (w @ fm).item()
                        r_shrunk[m] = (w_shrunk @ fm).item()

                        # Sharpe: Equal-weighted factor portfolio
                        r_eq[m] = (w_eq @ fm).item()

                        # Sharpe: Long-short equal-weighted (based on true returns)
                        pred_ret = (Bm @ mu_F.unsqueeze(1)).squeeze(1)
                        N = pred_ret.shape[0]
                        decile_size = max(N // 10, 1)
                        sorted_idx = pred_ret.argsort()
                        top10 = sorted_idx[-decile_size:]
                        bottom10 = sorted_idx[:decile_size]
                        r_ls[m] = (Ym[top10].mean() - Ym[bottom10].mean()).item()

                        # R2_Pred
                        Yhat_pred = (Bm @ mu_F.unsqueeze(1)).squeeze(1)
                        pred_sse += ((Ym - Yhat_pred) ** 2).sum().item()

            train_dates += val_dates[:train_freq]
            if OOS_window == 'rolling':
                train_dates = train_dates[-OOS_window_specs:]
                f_hats = {d: f_hats[d] for d in train_dates}
            idx0 += train_freq

        R2_tot = 1 - total_sse / total_sst
        R2_pred = 1 - pred_sse / total_sst

        def sharpe(r_dict):
            r = torch.tensor([r_dict[m] for m in sorted(r_dict)], dtype=torch.float32)
            return (r.mean() / r.std(unbiased=False)) * (12 ** 0.5)

        return {
            'R2_Total_OOS': R2_tot,
            'R2_Pred_OOS':  R2_pred,
            'Sharpe_Ratio': sharpe(r_p),
            'Sharpe_Shrunk': sharpe(r_shrunk),
            'Sharpe_LS': sharpe(r_ls)
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
