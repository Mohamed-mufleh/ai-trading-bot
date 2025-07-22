from dataset import prepare_features
from model import train_model
from backtest import run_backtest

# Parametry dataset
SEQ_LEN = 24
FUTURE_WINDOW = 5
BUY_THRESHOLD_REST = 0.015
SELL_THRESHOLD_REST = -0.015
BUY_THRESHOLD_2024_2025 = 0.015
SELL_THRESHOLD_2024_2025 = -0.015

# Parametry model
BATCH_SIZE = 64
EPOCHS = 500

symbol = 'BTCUSDT'
timeframe = '1h'
limit = 100000

epochs = EPOCHS
for i in range(400):
    # Co 5 iteracji zwiększ thresholdy o 0.01
    if i % 5 == 0 and i > 0:
        BUY_THRESHOLD_REST += 0.005
        SELL_THRESHOLD_REST -= 0.005
        BUY_THRESHOLD_2024_2025 += 0.005
        SELL_THRESHOLD_2024_2025 -= 0.005
    # Co 15 iteracji zmniejsz EPOCHS o 5 (nie mniej niż 45)
    if i % 10 == 0 and i > 0:
        epochs = max(45, epochs - 20)
    print(f"\n=== Iteracja {i+1} | BUY: {BUY_THRESHOLD_REST:.3f} | SELL: {SELL_THRESHOLD_REST:.3f} | EPOCHS: {epochs} ===")
    prepare_features(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        seq_len=SEQ_LEN,
        future_window=FUTURE_WINDOW,
        buy_threshold_rest=BUY_THRESHOLD_REST,
        sell_threshold_rest=SELL_THRESHOLD_REST,
        buy_threshold_2024_2025=BUY_THRESHOLD_2024_2025,
        sell_threshold_2024_2025=SELL_THRESHOLD_2024_2025
    )
    train_model(batch_size=BATCH_SIZE, epochs=epochs)
    run_backtest()