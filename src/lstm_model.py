"""
lstm_model.py
-------------
LSTM implemented from scratch using NumPy.
Architecture follows the 1xM (row-vector) convention used in coursework.

Gate equations (all vectors are 1 x K):
    i(t) = sigmoid( x(t)@U_i + h(t-1)@W_i + b_i )   # input gate
    f(t) = sigmoid( x(t)@U_f + h(t-1)@W_f + b_f )   # forget gate
    o(t) = sigmoid( x(t)@U_o + h(t-1)@W_o + b_o )   # output gate
    g(t) = tanh(    x(t)@U_g + h(t-1)@W_g          ) # new memory (cell input)
    c(t) = f(t) * c(t-1) + i(t) * g(t)               # cell state
    h(t) = o(t) * tanh(c(t))                          # hidden state
    y(t) = h(t) @ V + b_y                             # output (regression)

Parameter estimation: BFGS via scipy.optimize.minimize,
directly following LSTM_full_anatomy_1xM.ipynb.
"""

import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class FromScratchLSTM:
    """
    LSTM with BFGS-based parameter estimation (no deep learning framework).

    Parameters
    ----------
    input_size  : M  number of input features per timestep
    hidden_size : K  cell/hidden state dimension
    output_size : N  number of output values (default 1 for regression)
    """

    def __init__(self, input_size: int, hidden_size: int = 3, output_size: int = 1):
        self.M = input_size
        self.K = hidden_size
        self.N = output_size
        self.params_ = None

        MK = self.M * self.K
        KK = self.K * self.K
        K  = self.K

        # Parameter count:
        # 4 gates × (U: MK + W: KK + b: K) + output (V: KN + b_y: N)
        self._n_params = 4 * (MK + KK + K) + self.K * self.N + self.N

    # ── private helpers ──────────────────────────────────────────────────

    def _unpack(self, p: np.ndarray) -> tuple:
        """Unpack flat vector into weight matrices for all gates."""
        M, K, N = self.M, self.K, self.N
        MK, KK = M * K, K * K
        idx = 0

        def _take(n):
            nonlocal idx
            out = p[idx:idx + n]
            idx += n
            return out

        U_i = _take(MK).reshape(M, K)
        W_i = _take(KK).reshape(K, K)
        b_i = _take(K).reshape(1, K)

        U_f = _take(MK).reshape(M, K)
        W_f = _take(KK).reshape(K, K)
        b_f = _take(K).reshape(1, K)

        U_o = _take(MK).reshape(M, K)
        W_o = _take(KK).reshape(K, K)
        b_o = _take(K).reshape(1, K)

        U_g = _take(MK).reshape(M, K)
        W_g = _take(KK).reshape(K, K)
        # no bias for new-memory gate (matches coursework convention)

        V   = _take(K * N).reshape(K, N)
        b_y = _take(N).reshape(1, N)

        return U_i, W_i, b_i, U_f, W_f, b_f, U_o, W_o, b_o, U_g, W_g, V, b_y

    def _forward(self, param: np.ndarray, inputs, targets) -> float:
        """Forward pass; returns RMSE loss."""
        (U_i, W_i, b_i,
         U_f, W_f, b_f,
         U_o, W_o, b_o,
         U_g, W_g,
         V, b_y) = self._unpack(param)

        h = np.zeros((1, self.K))
        c = np.zeros((1, self.K))
        loss = 0.0
        n = 0

        for t, x_row in inputs.iterrows():
            x = np.array(x_row).reshape(1, -1)                     # 1 x M

            i_gate = _sigmoid(x @ U_i + h @ W_i + b_i)            # 1 x K
            f_gate = _sigmoid(x @ U_f + h @ W_f + b_f)            # 1 x K
            o_gate = _sigmoid(x @ U_o + h @ W_o + b_o)            # 1 x K
            g_gate = np.tanh(x @ U_g + h @ W_g)                   # 1 x K

            c = f_gate * c + i_gate * g_gate                       # 1 x K
            h = o_gate * np.tanh(c)                                # 1 x K

            y_hat = (h @ V + b_y).flatten()[0]                     # scalar
            loss += (targets[t] - y_hat) ** 2
            n += 1

        return np.sqrt(loss / n)   # RMSE (matches coursework)

    # ── public API ───────────────────────────────────────────────────────

    def fit(self, inputs, targets, tol: float = 1e-3, verbose: bool = True):
        """
        Estimate all LSTM parameters by minimising RMSE with BFGS.

        Parameters
        ----------
        inputs  : pd.DataFrame  shape (T, M)
        targets : pd.Series     shape (T,)
        tol     : float         convergence tolerance
        verbose : bool          print BFGS progress
        """
        np.random.seed(42)
        param0 = np.random.randn(self._n_params) * 0.01

        logger.info(
            f"[LSTM] Starting BFGS optimisation "
            f"({self._n_params} parameters, {len(inputs)} training samples)..."
        )
        result = minimize(
            self._forward,
            param0,
            args=(inputs, targets),
            method="BFGS",
            tol=tol,
            options={"disp": verbose}
        )
        self.params_ = result.x
        logger.info(f"[LSTM] Optimisation complete. Final RMSE: {result.fun:.6f}")
        return self

    def predict(self, inputs) -> np.ndarray:
        """
        Run forward pass with fitted parameters; returns predictions array.
        """
        if self.params_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        (U_i, W_i, b_i,
         U_f, W_f, b_f,
         U_o, W_o, b_o,
         U_g, W_g,
         V, b_y) = self._unpack(self.params_)

        h = np.zeros((1, self.K))
        c = np.zeros((1, self.K))
        preds = []

        for _, x_row in inputs.iterrows():
            x = np.array(x_row).reshape(1, -1)
            i_gate = _sigmoid(x @ U_i + h @ W_i + b_i)
            f_gate = _sigmoid(x @ U_f + h @ W_f + b_f)
            o_gate = _sigmoid(x @ U_o + h @ W_o + b_o)
            g_gate = np.tanh(x @ U_g + h @ W_g)
            c = f_gate * c + i_gate * g_gate
            h = o_gate * np.tanh(c)
            y_hat = (h @ V + b_y).flatten()[0]
            preds.append(y_hat)

        return np.array(preds)


# ── smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_pipeline import build_pipeline

    i_train, i_test, t_train, t_test, scaler, _ = build_pipeline(
        "AAPL", start="2023-01-01", end="2024-01-01"
    )

    lstm = FromScratchLSTM(input_size=i_train.shape[1], hidden_size=3)
    lstm.fit(i_train, t_train, tol=1e-2)

    preds = lstm.predict(i_test)
    rmse = np.sqrt(np.mean((t_test.values - preds) ** 2))
    print(f"\nTest RMSE (scaled): {rmse:.4f}")
