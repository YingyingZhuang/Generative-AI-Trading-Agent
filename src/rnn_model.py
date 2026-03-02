"""
rnn_model.py
------------
Vanilla RNN implemented from scratch using NumPy.
Architecture follows the 1xM (row-vector) convention used in coursework.

Forward pass equation:
    h(t) = tanh( x(t) @ U  +  h(t-1) @ W  +  b )
    y(t) = h(t) @ V  +  c

Parameter estimation via scipy.optimize.minimize (BFGS),
matching the approach in RNN_anatomy_stocks.ipynb.
"""

import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class VanillaRNN:
    """
    Vanilla RNN with BFGS-based parameter estimation.

    Parameters
    ----------
    input_size  : M  (number of input features per timestep)
    hidden_size : K  (hidden state dimension)
    output_size : N  (number of output values, default 1 for regression)
    """

    def __init__(self, input_size: int, hidden_size: int = 3, output_size: int = 1):
        self.M = input_size
        self.K = hidden_size
        self.N = output_size
        self.params_ = None   # set after fit()

        # Count total parameters for BFGS vector
        # U: M x K, W: K x K, b: 1 x K, V: K x N, c: 1 x N
        self._n_params = (
            self.M * self.K      # U
            + self.K * self.K    # W
            + self.K             # b
            + self.K * self.N    # V
            + self.N             # c
        )

    # ── private helpers ──────────────────────────────────────────────────

    def _unpack(self, param: np.ndarray) -> tuple:
        """Unpack flat parameter vector into weight matrices."""
        M, K, N = self.M, self.K, self.N
        idx = 0

        U  = param[idx:idx + M*K].reshape(M, K);   idx += M*K
        W  = param[idx:idx + K*K].reshape(K, K);   idx += K*K
        b  = param[idx:idx + K].reshape(1, K);     idx += K
        V  = param[idx:idx + K*N].reshape(K, N);   idx += K*N
        c  = param[idx:idx + N].reshape(1, N);     idx += N

        return U, W, b, V, c

    def _forward(self, param: np.ndarray, inputs, targets) -> float:
        """Forward pass; returns MSE loss."""
        U, W, b, V, c = self._unpack(param)

        h_prev = np.zeros((1, self.K))
        loss = 0.0
        n = 0

        for t, x_row in inputs.iterrows():
            x = np.array(x_row).reshape(1, -1)          # 1 x M
            h = np.tanh(x @ U + h_prev @ W + b)         # 1 x K
            y_hat = (h @ V + c).flatten()[0]             # scalar
            loss += (targets[t] - y_hat) ** 2
            h_prev = h
            n += 1

        return loss / n

    # ── public API ───────────────────────────────────────────────────────

    def fit(self, inputs, targets, tol: float = 1e-3, verbose: bool = True):
        """
        Estimate parameters by minimising MSE with BFGS.

        Parameters
        ----------
        inputs  : pd.DataFrame  shape (T, M)
        targets : pd.Series     shape (T,)
        tol     : float         convergence tolerance passed to scipy.minimize
        verbose : bool          print optimizer output
        """
        np.random.seed(42)
        param0 = np.random.randn(self._n_params) * 0.01

        logger.info(f"[RNN] Starting BFGS optimisation ({self._n_params} parameters)...")
        result = minimize(
            self._forward,
            param0,
            args=(inputs, targets),
            method="BFGS",
            tol=tol,
            options={"disp": verbose}
        )
        self.params_ = result.x
        logger.info(f"[RNN] Optimisation complete. Final MSE: {result.fun:.6f}")
        return self

    def predict(self, inputs) -> np.ndarray:
        """
        Run forward pass with fitted parameters; returns predictions array.
        """
        if self.params_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        U, W, b, V, c = self._unpack(self.params_)
        h_prev = np.zeros((1, self.K))
        preds = []

        for _, x_row in inputs.iterrows():
            x = np.array(x_row).reshape(1, -1)
            h = np.tanh(x @ U + h_prev @ W + b)
            y_hat = (h @ V + c).flatten()[0]
            preds.append(y_hat)
            h_prev = h

        return np.array(preds)


# ── smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_pipeline import build_pipeline

    i_train, i_test, t_train, t_test, scaler, _ = build_pipeline(
        "TSLA", start="2023-01-01", end="2024-01-01"
    )

    rnn = VanillaRNN(input_size=i_train.shape[1], hidden_size=3)
    rnn.fit(i_train, t_train, tol=1e-2)

    preds = rnn.predict(i_test)
    rmse = np.sqrt(np.mean((t_test.values - preds) ** 2))
    print(f"\nTest RMSE (scaled): {rmse:.4f}")
