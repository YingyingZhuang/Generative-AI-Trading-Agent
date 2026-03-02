"""
train.py
--------
Main entry point for FinSignal.

Runs the full pipeline:
    1. Fetch and preprocess data  (data_pipeline.py)
    2. Train RNN from scratch     (rnn_model.py)
    3. Train LSTM from scratch    (lstm_model.py)
    4. Evaluate and compare       (evaluate.py)
    5. Save prediction plot       (results/)

Usage:
    python train.py                          # defaults: AAPL, 2020-2024
    python train.py --ticker TSLA            # different ticker
    python train.py --ticker META --tol 1e-2 # looser convergence (faster)
"""

import argparse
import numpy as np
import sys
import os

# Allow running from repo root or from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline import build_pipeline
from rnn_model     import VanillaRNN
from lstm_model    import FromScratchLSTM
from evaluate      import print_metrics, plot_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="FinSignal training script")
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Yahoo Finance ticker symbol (default: AAPL)")
    parser.add_argument("--start",  type=str, default="2020-01-01")
    parser.add_argument("--end",    type=str, default="2024-01-01")
    parser.add_argument("--hidden", type=int, default=3,
                        help="Hidden state size K (default: 3)")
    parser.add_argument("--tol",    type=float, default=1e-2,
                        help="BFGS convergence tolerance (default: 1e-2)")
    parser.add_argument("--no-lstm", action="store_true",
                        help="Skip LSTM training (faster smoke-test)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"  FinSignal: RNN/LSTM From-Scratch Forecasting Pipeline")
    print(f"  Ticker: {args.ticker}  |  {args.start} → {args.end}")
    print("=" * 60)

    # ── 1. Data pipeline ──────────────────────────────────────────────
    print("\n[1/4] Running ETL pipeline...")
    i_train, i_test, t_train, t_test, scaler, df = build_pipeline(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
    )
    print(f"      Input features : {list(i_train.columns)}")
    print(f"      Training size  : {len(i_train)}")
    print(f"      Test size      : {len(i_test)}")

    # ── 2. Train RNN ──────────────────────────────────────────────────
    print("\n[2/4] Training Vanilla RNN (from scratch, BFGS)...")
    rnn = VanillaRNN(input_size=i_train.shape[1], hidden_size=args.hidden)
    rnn.fit(i_train, t_train, tol=args.tol, verbose=True)
    rnn_preds = rnn.predict(i_test)

    print("\n  >> RNN Test Metrics:")
    rnn_rmse, rnn_ic = print_metrics(t_test.values, rnn_preds, label="RNN")

    predictions = {"Vanilla RNN": rnn_preds}

    # ── 3. Train LSTM ─────────────────────────────────────────────────
    if not args.no_lstm:
        print("\n[3/4] Training From-Scratch LSTM (BFGS)...")
        lstm = FromScratchLSTM(input_size=i_train.shape[1], hidden_size=args.hidden)
        lstm.fit(i_train, t_train, tol=args.tol, verbose=True)
        lstm_preds = lstm.predict(i_test)

        print("\n  >> LSTM Test Metrics:")
        lstm_rmse, lstm_ic = print_metrics(t_test.values, lstm_preds, label="LSTM")

        predictions["From-Scratch LSTM"] = lstm_preds
    else:
        print("\n[3/4] LSTM skipped (--no-lstm flag).")

    # ── 4. Evaluate & plot ────────────────────────────────────────────
    print("\n[4/4] Generating prediction plot...")
    plot_path = plot_predictions(
        y_true=t_test.values,
        predictions=predictions,
        ticker=args.ticker,
        save_dir="results"
    )

    # ── 5. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Ticker : {args.ticker}  |  Test samples: {len(i_test)}")
    print(f"  RNN  → RMSE: {rnn_rmse:.4f}  |  IC: {rnn_ic:.4f}")
    if not args.no_lstm:
        print(f"  LSTM → RMSE: {lstm_rmse:.4f}  |  IC: {lstm_ic:.4f}")
    print(f"\n  Plot saved: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
