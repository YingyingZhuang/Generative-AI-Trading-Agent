# Generative-AI-Trading-Agent
# LynchBot: Generative AI Financial Agent
This project is an end-to-end Machine Learning pipeline that integrates quantitative time series forecasting with qualitative investment strategy extraction, inspired by Peter Lynch's philosophy.

*(Note: This repository is actively under development as an MVP portfolio project. Core deep learning models are currently available in exploratory Jupyter Notebooks, and the pipeline is undergoing refactoring into modular Python scripts for UI integration.)*

## Project Vision & Architecture

The objective of **LynchBot** is to bridge the gap between numerical stock data and natural language financial strategies using a hybrid AI approach. 

The planned architecture consists of three core engineering pillars:
1. **Data Engineering Pipeline:** Automated ingestion and preprocessing of unstructured financial texts (tokenization, padding) and structured OHLCV stock data (MinMaxScaler, sequence generation).
2. **Predictive Engine (LSTM):** Deep learning neural networks utilizing Keras/TensorFlow to predict financial text sequences and analyze noisy time-series data.
3. **Generative Interface:** A deployment-ready Streamlit frontend to interact with the trained models via natural language queries.

## Tech Stack
* **Deep Learning Frameworks:** TensorFlow, Keras (LSTM, Embedding, Dense layers)
* **Data Engineering:** Python, NumPy, Pandas, Scikit-learn
* **NLP Techniques:** N-gram Sequencing, Categorical Cross-Entropy, Tokenization

## Current Repository Contents

### 1. NLP Text Generation MVP (`LSTM_Lynch_Text_Generation_MVP.ipynb`)
This notebook demonstrates the foundational Natural Language Processing (NLP) pipeline for generating financial text.
* **Feature Engineering:** Implemented a robust preprocessing pipeline using Keras `Tokenizer` and `pad_sequences` to convert unstructured Peter Lynch quotes into padded N-gram sequences.
* **Model Architecture:** Built a Sequential neural network featuring an `Embedding` layer (64 dimensions), an `LSTM` layer (50 units), and a `Dense` output layer utilizing a `softmax` activation function for multi-class sequence prediction.
* **Training & Optimization:** Trained the model using the Adam optimizer and categorical cross-entropy loss function to predict the next contextual word in a financial sequence.

## Next Steps / Roadmap
* **Script Modularization:** Extract data loading, model definition, and training loops into separate `.py` modules (`data_loader.py`, `model.py`, `train.py`) to adhere to software engineering best practices.
* **Time-Series Integration:** Introduce the OHLCV stock data forecasting module, highlighting hyperparameter tuning (e.g., optimizing batch sizes to mitigate overfitting on highly volatile tech stocks like META).
* **UI Deployment:** Wrap the inference logic into a Streamlit application for interactive demonstrations.