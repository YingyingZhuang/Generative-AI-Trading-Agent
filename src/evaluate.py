"""
evaluate.py
-----------
Model evaluation utilities:
- RMSE
- Information Coefficient (IC) via Spearman rank correlation
- Prediction vs actual plot saved to results/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Information Coefficient: Spearman rank correlation between
    predicted and actual values.  Range [-1, 1]; higher is better.
    IC > 0.05 is considered meaningful in finance.
    """
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic)


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = ""):
    """Print RMSE and IC to stdout."""
    rmse = compute_rmse(y_true, y_pred)
    ic   = compute_ic(y_true, y_pred)
    tag  = f"[{label}] " if label else ""
    print(f"{tag}RMSE: {rmse:.4f}  |  IC: {ic:.4f}")
    return rmse, ic


def plot_predictions(
    y_true: np.ndarray,
    predictions: dict,          # {"RNN": array, "LSTM": array, ...}
    ticker: str = "AAPL",
    save_dir: str = "results"
) -> str:
    """
    Plot actual vs predicted values for one or more models.

    Parameters
    ----------
    y_true      : ground truth array
    predictions : dict mapping model name → prediction array
    ticker      : stock ticker (used in title)
    save_dir    : directory to save the PNG

    Returns
    -------
    str  path to saved PNG file
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(y_true, label="Actual", color="black", linewidth=1.5, alpha=0.8)

    colours = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    for (name, preds), colour in zip(predictions.items(), colours):
        rmse = compute_rmse(y_true, preds)
        ic   = compute_ic(y_true, preds)
        ax.plot(
            preds,
            label=f"{name}  (RMSE={rmse:.4f}, IC={ic:.4f})",
            color=colour,
            linewidth=1.2,
            linestyle="--",
            alpha=0.85
        )

    ax.set_title(f"{ticker} — Next-Day Close Price: Actual vs Predicted (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Day (Test Period)", fontsize=11)
    ax.set_ylabel("Normalised Close Price", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, f"{ticker}_prediction_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EVALUATE] Plot saved → {path}")
    return path


# ── smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    y_true = np.random.rand(50)
    y_pred = y_true + np.random.randn(50) * 0.05
    print_metrics(y_true, y_pred, label="Demo")
    plot_predictions(y_true, {"Demo Model": y_pred}, ticker="TEST")
