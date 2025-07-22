# 🤖 AI Crypto Trading Bot – Deep Learning with 3D CNN + GRU

A deep learning–powered crypto trading bot using a hybrid 3D CNN + GRU model to predict market signals (`BUY`, `SELL`, `HOLD`) and execute real-time trades via the Binance API.

---

## 🧠 Key Features

- 🔍 **Hybrid Deep Learning Model**  
  Combines 3D Convolutional Neural Networks and GRU layers to analyze spatio-temporal patterns in market data

- 🧾 **Real-Time Trading**  
  Executes live trades using the Binance API, based on the model’s predictions

- 🛠️ **Customizable Parameters**  
  Easily tune training parameters like `epochs`, `batch size`, `learning rate`, etc.

- 💾 **Model State Management**  
  Saves and loads the best-performing model via PyTorch's `state_dict`

- 💹 **Backtesting Module**  
  Run strategy simulations on historical data to evaluate performance

- 🔗 **Binance Integration**  
  Fetches real-time market data (default: BTC/USDT) with easy support for other pairs

- 📦 **Modular Codebase**  
  Designed for clarity and experimentation — each stage is separated and reusable

---

## ⚙️ How It Works

The pipeline consists of five core Python scripts, each responsible for a key step in the trading workflow:

1. **Data Preparation**  
   Collects raw crypto data from Binance, computes technical indicators, assigns labels (`BUY`, `SELL`, `HOLD`), normalizes inputs, and applies oversampling to balance the dataset.

2. **Model Training**  
   Defines and trains the hybrid deep learning model using a combination of 3D CNN and GRU layers. Includes the training loop, loss tracking, and model saving.

3. **Backtesting**  
   Tests the trained model on historical data to estimate profitability and performance over a selected time period. Outputs include trade logs and profit metrics.

4. **Dynamic Optimizer**  
   Connects data prep, training, and backtesting in one loop. Automatically adjusts parameters (e.g., learning rate, batch size) every N iterations to search for better results.

5. **Live Trading Bot**  
   Uses the trained model to make predictions in real-time and places trades via the Binance API based on the current market conditions.

---
