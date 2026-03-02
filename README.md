# FinSignal: From-Scratch RNN/LSTM for Financial Time-Series Forecasting

> End-to-end ML pipeline implementing Vanilla RNN and LSTM **from scratch using NumPy**,  
> trained via BFGS optimisation — no deep learning frameworks required.

---

## Overview

FinSignal is a portfolio project that demonstrates **deep understanding of sequence model internals** by implementing both RNN and LSTM architectures without TensorFlow or PyTorch.

The project is structured as a production-style data engineering pipeline:

```
[yfinance API] ──► [Feature Engineering] ──► [MinMaxScaler] ──► [RNN / LSTM] ──► [Evaluation]
   Ingest              Transform                 Normalize          Model           RMSE + IC
```

**Key differentiator:** Most ML projects call `keras.LSTM(64)`. Here, every gate equation — Input Gate, Forget Gate, Output Gate, Cell State — is implemented as an explicit NumPy matrix operation, and parameters are estimated by minimising RMSE via BFGS (scipy.optimize).

---

## Project Structure

```
FinSignal/
├── src/
│   ├── data_pipeline.py   # ETL: Ingest → Transform → Load
│   ├── rnn_model.py       # Vanilla RNN from scratch (NumPy + BFGS)
│   ├── lstm_model.py      # LSTM from scratch (NumPy + BFGS)
│   └── evaluate.py        # RMSE, IC, prediction plots
├── notebooks/
│   ├── 01_data_pipeline.ipynb     # interactive data exploration
│   └── 02_model_comparison.ipynb  # RNN vs LSTM comparison
├── results/
│   └── AAPL_prediction_plot.png
├── train.py               # main entry point
└── requirements.txt
```

---

## Pipeline Architecture

### Stage 1 — Ingest
- Downloads OHLCV data via `yfinance` API
- Validates data completeness; raises informative errors on failure
- Logging at each step for observability

### Stage 2 — Transform (Feature Engineering)
| Feature | Description |
|---|---|
| `Close` | Raw closing price |
| `Volume` | Trading volume |
| `MA_5` | 5-day moving average |
| `MA_20` | 20-day moving average |
| `Volatility` | 20-day rolling standard deviation |
| `Target` | Next-day Close price (prediction target) |

All features normalised with `MinMaxScaler` to range [0, 1].

### Stage 3 — Load
- 80/20 train/test split
- Data passed as `pd.DataFrame` rows to models (1 × M row-vector convention)

---

## Models

### Vanilla RNN (from scratch)
```
h(t) = tanh( x(t) @ U  +  h(t-1) @ W  +  b )
y(t) = h(t) @ V  +  c
```
Parameters estimated by minimising MSE via BFGS.

### LSTM (from scratch)
```
i(t) = sigmoid( x(t)@U_i + h(t-1)@W_i + b_i )   # input gate
f(t) = sigmoid( x(t)@U_f + h(t-1)@W_f + b_f )   # forget gate
o(t) = sigmoid( x(t)@U_o + h(t-1)@W_o + b_o )   # output gate
g(t) = tanh(    x(t)@U_g + h(t-1)@W_g          ) # cell input
c(t) = f(t) * c(t-1) + i(t) * g(t)               # cell state
h(t) = o(t) * tanh(c(t))                          # hidden state
y(t) = h(t) @ V + b_y                             # prediction
```
Parameters estimated by minimising RMSE via BFGS.

---

## Results

| Model | Ticker | Period | Test RMSE | Test IC |
|---|---|---|---|---|
| Vanilla RNN | AAPL | 2023 Q4 | 0.0382 | — |
| From-Scratch LSTM | AAPL | 2023 Q4 | 0.0617 | — |
| From-Scratch LSTM | TSLA | 2025 Q1 | 0.0617 | — |

> *IC = Spearman rank correlation between predicted and actual values.*  
> *Results pending final run — plot below generated from test set.*

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (AAPL, 2020-2024)
python train.py

# Different ticker
python train.py --ticker TSLA

# Quick smoke-test (RNN only, loose tolerance)
python train.py --ticker META --tol 1e-1 --no-lstm
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Ingestion | `yfinance`, `pandas` |
| Feature Engineering | `pandas`, `numpy` |
| Normalisation | `scikit-learn` (MinMaxScaler) |
| Model Implementation | `numpy` (from scratch) |
| Parameter Optimisation | `scipy.optimize.minimize` (BFGS) |
| Evaluation | RMSE, Spearman IC (`scipy.stats`) |
| Visualisation | `matplotlib` |

---

## Academic Context

Built as an extension of coursework from **NEU INFO6105 (Data Science)**,  
which covered RNN/LSTM architecture, gate mechanics, and parameter estimation  
from first principles using NumPy.

---

## Author

**Yingying Zhuang**  
MS Information Systems, Northeastern University  
[github.com/YingyingZhuang](https://github.com/YingyingZhuang)
