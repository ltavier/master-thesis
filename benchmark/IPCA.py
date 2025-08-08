import pandas as pd
import numpy as np
import scipy.linalg as sla
from sklearn.covariance import LedoitWolf
from scipy.stats import norm

# matrix left/right division
_mldivide = lambda A, B: sla.lstsq(np.array(A), np.array(B))[0]
_mrdivide = lambda A, B: sla.lstsq(np.array(B).T, np.array(A).T)[0].T

class SimpleIPCA:
    def __init__(self, RZ, return_column=0, add_constant=True):
        """
        Simplified IPCA: always latent factors only (gFac=None), Rdo=True, Betado=False.
        """
        # Split returns and characteristics
        if isinstance(return_column, int):
            self.R = RZ.iloc[:, return_column].to_frame()
        else:
            self.R = RZ.loc[:, return_column].to_frame()
        self.Z = RZ.drop(columns=self.R.columns)

        # Build X and W
        chars = list(self.Z.columns)
        if add_constant:
            chars.append('Constant')
        dates = list(self.Z.index.get_level_values(0).unique())
        self.X = pd.DataFrame(index=chars, columns=dates, dtype=float)
        self.W = pd.DataFrame(
            index=pd.MultiIndex.from_product([dates, chars], names=['date', 'char']),
            columns=chars,
            dtype=float
        )
        self.Nts = pd.Series(index=dates, dtype=int)

        for d in dates:
            Zd = self.Z.loc[d].values
            if add_constant:
                ones = np.ones((Zd.shape[0], 1))
                Zd = np.hstack([Zd, ones])

            self.Nts[d] = Zd.shape[0]
            returns = self.R.loc[d].values.flatten()
            self.X[d] = Zd.T.dot(returns) / self.Nts[d]
            self.W.loc[d] = Zd.T.dot(Zd) / self.Nts[d]

        # Store for speed
        self.dates = dates
        self._X = self.X.values
        L, T = self._X.shape
        self._W = np.zeros((L, L, T), dtype=float)

        for i, d in enumerate(dates):
            matrix = self.W.loc[d].values.reshape(L, L)
            self._W[:, :, i] = matrix

        self.chars = chars
        self.L = len(chars)

    def _svd_initial(self, K):
        """ SVD init for latent factors """
        U, s, VT = sla.svd(self.X.values, full_matrices=False)
        Gamma = U[:, :K]
        Factor = np.diag(s[:K]).dot(VT[:K, :])
        return Gamma, Factor

    def _normalize(self, Gamma, Factor, choice, specs):
        """ Apply normalization to Gamma and Factor """
        if choice == 'PCA_positivemean':
            R1 = sla.cholesky(Gamma.T.dot(Gamma))
            U2, _, _ = sla.svd(R1.dot(Factor).dot(Factor.T).dot(R1.T))
            Gamma_n = _mrdivide(Gamma, R1).dot(U2)
            Factor_n = _mldivide(U2, R1.dot(Factor))
            signs = np.sign(Factor_n.mean(axis=1)).reshape(-1, 1)
            signs[signs == 0] = 1
            Gamma_n = Gamma_n * signs.T
            Factor_n = Factor_n * signs
            return Gamma_n, Factor_n

        if choice == 'Identity':
            Rmat = Gamma[specs]
            Gamma_n = _mrdivide(Gamma, Rmat)
            Factor_n = _mldivide(Rmat, Factor)
            return Gamma_n, Factor_n

        raise ValueError(f'Unknown normalization choice: {choice}')

    def _linear_als(self, Gamma, Factor, K, normalization_choice, normalization_specs):
        """ One ALS iteration with normalization option """
        T = len(self.dates)
        Fnew = np.zeros_like(Factor)

        for i in range(T):
            Wd = self._W[:, :, i]
            Xd = self._X[:, i]
            lhs = Gamma.T.dot(Wd).dot(Gamma)
            rhs = Gamma.T.dot(Xd)
            Fnew[:, i] = _mldivide(lhs, rhs)

        numer = np.zeros(self.L * K)
        denom = np.zeros((self.L * K, self.L * K))

        for i, d in enumerate(self.dates):
            x = self._X[:, i]
            f = Fnew[:, i]
            numer += np.kron(x, f) * self.Nts[d]
            denom += np.kron(self._W[:, :, i], np.outer(f, f)) * self.Nts[d]

        Gnew = np.reshape(_mldivide(denom, numer), (self.L, K))
        return self._normalize(Gnew, Fnew, normalization_choice, normalization_specs)

    def fit(
        self,
        K=1,
        minTol=1e-1,
        maxIters=500,
        normalization_choice='PCA_positivemean',
        normalization_specs=None
    ):
        """
        Estimate latent factors, returning weights, mu_F, mu_shrunk, etc.
        """
        Gamma, Factor = self._svd_initial(K)
        tol = np.inf
        it = 0

        while it < maxIters and tol > minTol:
            Gnew, Fnew = self._linear_als(
                Gamma,
                Factor,
                K,
                normalization_choice,
                normalization_specs
            )
            tol = max(
                np.max(np.abs(Gnew - Gamma)),
                np.max(np.abs(Fnew - Factor))
            )
            Gamma = Gnew
            Factor = Fnew
            it += 1

        return {
            'Gamma': Gamma,
            'Factor': Factor
        }
    def compute_tangency_weights(
        self,
        factor_ts,
        target_vol=None
    ):
        """
        Compute tangency portfolio weights using both the empirical covariance and Ledoit-Wolf shrinkage.

        Accepts factor_ts as a (K, T) array or dict of 1D arrays (length T).

        Returns:
            w_emp: Tangency weights from empirical covariance.    # [K]
            w_lw:  Tangency weights from Ledoit-Wolf covariance.  # [K]
            mu_F:  Sample mean of factor returns.                # [K]
        """
        # 0) Handle dict→array
        if isinstance(factor_ts, dict):
            # assume each dict entry is a 1D array of length T
            factor_ts = np.column_stack([factor_ts[k] for k in sorted(factor_ts)])
        # now factor_ts shape = (K, T)
        K, T = factor_ts.shape

        # 1) Sample mean per factor
        mu_F = factor_ts.mean(axis=1)               # [K]

        # 2a) Empirical covariance estimate
        X = factor_ts - mu_F[:, None]               # [K, T] centered
        Sigma_emp = X @ X.T / (T - 1)               # [K, K]

        # 2b) Ledoit-Wolf covariance estimate
        lw = LedoitWolf().fit(factor_ts.T)          # expects shape (T, K)
        Sigma_lw = lw.covariance_                   # [K, K]

        # 3a) Raw tangency weights (empirical)
        w_emp = np.linalg.solve(Sigma_emp, mu_F)    # [K]
        # 3b) Raw tangency weights (Ledoit-Wolf)
        w_lw = np.linalg.solve(Sigma_lw, mu_F)      # [K]

        # 4) Scale to target vol or sum to 1
        def scale_weights(w, Sigma):
            if target_vol is not None:
                vol = np.sqrt(w @ Sigma @ w).clip(min=1e-12)
                return w * (target_vol / vol)
            else:
                return w / w.sum()

        w_emp = scale_weights(w_emp, Sigma_emp)
        w_lw = scale_weights(w_lw, Sigma_lw)

        return w_emp, w_lw, mu_F


    def fit_OOS(
        self,
        K=1,
        OOS_window='recursive',
        OOS_window_specs=60,
        train_freq=12,
        normalization_choice='PCA_positivemean',
        normalization_specs=None
    ):
        """
        OOS evaluation without validation set.
        Returns:
        - R2 metrics
        - Sharpe_Ratio         (tangency portfolio)
        - Sharpe_EW_Factors    (equal-weighted factor portfolio)
        - Sharpe_LS            (equal-weighted long-short portfolio)
        """
        Dates = list(self.dates)
        idx0 = OOS_window_specs
        total_sse = 0.0
        pred_sse = 0.0
        pred_sse_sig = 0.0
        total_sst = 0.0
        r_p = {}
        r_shrunk = {}
        r_ls = {}
        hist_facs = {}
        train_dates = Dates[:idx0].copy()

        orig = (
            self.dates,
            self.X.copy(),
            self.W.copy(),
            self.Nts.copy(),
            self._X.copy(),
            self._W.copy()
        )

        while idx0 + 2 * train_freq <= len(Dates):
            print(f"Train = {Dates[idx0-1]}, Val = {Dates[idx0 + train_freq-1]}, Test = {Dates[idx0 + 2 * train_freq-1]}")
            val_dates = Dates[idx0: idx0 + train_freq]
            test_dates = Dates[idx0 + train_freq:idx0 + 2 * train_freq]

            idxs = [Dates.index(d) for d in train_dates]
            self.dates = train_dates
            self.X = self.X[train_dates]
            self.W = self.W.loc[train_dates]
            self.Nts = self.Nts[train_dates]
            self._X = self.X.values
            self._W = orig[5][:, :, idxs]

            res = self.fit(
                K,
                normalization_choice=normalization_choice,
                normalization_specs=normalization_specs
            )

            Gamma = res['Gamma']
            Factor = res['Factor']

            self.dates, self.X, self.W, self.Nts, self._X, self._W = orig

            for i, d in enumerate(train_dates):
                hist_facs[d] = Factor[:, i]

            """
            for d in val_dates:
                i = Dates.index(d)
                Zd = self.Z.loc[d].values
                if Zd.ndim == 1:
                    Zd = Zd.reshape(-1, 1)
                if self.L > Zd.shape[1]:
                    Zd = np.hstack([Zd, np.ones((Zd.shape[0], 1))])
                Betat = Zd.dot(Gamma)
                fac_d = _mldivide(
                    Gamma.T.dot(self._W[:, :, i]).dot(Gamma),
                    Gamma.T.dot(self._X[:, i])
                )
                hist_facs[d] = fac_d
            """
                
            w, w_shrunk, mu_F = self.compute_tangency_weights(hist_facs,  target_vol=(12 ** 0.5) * 0.1)
            w_eq = np.ones_like(w) / w.size

            for d in test_dates:
                i = Dates.index(d)
                Yd = self.R.loc[d].values.flatten()
                Zd = self.Z.loc[d].values
                if Zd.ndim == 1:
                    Zd = Zd.reshape(-1, 1)
                if self.L > Zd.shape[1]:
                    Zd = np.hstack([Zd, np.ones((Zd.shape[0], 1))])

                Betat = Zd.dot(Gamma)
                fac_d = _mldivide(
                    Gamma.T.dot(self._W[:, :, i]).dot(Gamma),
                    Gamma.T.dot(self._X[:, i])
                )

                Yhat_in = Betat.dot(fac_d)
                total_sse += ((Yd - Yhat_in) ** 2).sum()
                total_sst += (Yd ** 2).sum()

                Yhat_pred = Betat.dot(mu_F)
                pred_sse += ((Yd - Yhat_pred) ** 2).sum()

                r_p[d] = w.dot(fac_d)
                r_shrunk[d] = w_shrunk.dot(fac_d)

                # Long-short portfolio:
                sorted_idx = np.argsort(Yhat_pred)
                N = len(Yhat_pred)
                decile_size = max(N // 10, 1)
                top_idx = sorted_idx[-decile_size:]
                bottom_idx = sorted_idx[:decile_size]
                r_ls[d] = Yd[top_idx].mean() - Yd[bottom_idx].mean()

                hist_facs[d] = fac_d

            train_dates.extend(val_dates[:train_freq])
            if OOS_window == 'rolling':
                train_dates = train_dates[-OOS_window_specs:]
                hist_facs = {d: hist_facs[d] for d in train_dates}
            idx0 += train_freq

        R2_tot = 1 - total_sse / total_sst
        R2_pred = 1 - pred_sse / total_sst

        def compute_sharpe(r_dict):
            r = np.array([r_dict[d] for d in sorted(r_dict)])
            return (r.mean() / r.std(ddof=0)) * np.sqrt(12)

        return {
            'R2_Total_OOS': R2_tot,
            'R2_Pred_OOS': R2_pred,
            'Sharpe_Ratio': compute_sharpe(r_p),
            'Sharpe_Shrunk': compute_sharpe(r_shrunk),
            'Sharpe_LS': compute_sharpe(r_ls),
        }
