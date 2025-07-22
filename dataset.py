from binance.client import Client
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ta
import numpy as np

# Parametry do łatwej zmiany w jednym miejscu
TF = "1h"
LIMIT = 100000
SEQ_LEN = 24
FUTURE_WINDOW = 5
BUY_THRESHOLD_REST = 0.055
SELL_THRESHOLD_REST = -0.055
BUY_THRESHOLD_2024_2025 = 0.055
SELL_THRESHOLD_2024_2025 = -0.055

def fetch_binance_ohlcv(symbol='BTCUSDT', timeframe=TF, limit=LIMIT):
    client = Client()
    klines = client.get_historical_klines(symbol, timeframe, f"{limit} hours ago UTC")
    df = pd.DataFrame(klines, columns=[
        'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Taker buy base asset volume', 'Taker buy quote asset volume']]
    
    # Convert to numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume',
                'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def add_indicators(df):
    # --- Trend ---
    df['ema_9'] = ta.trend.EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(close=df['Close'], window=21).ema_indicator()
    df['ema_trend'] = df['ema_9'] - df['ema_21']
    #df['ema_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    #df['ema_200'] = ta.trend.EMAIndicator(close=df['Close'], window=200).ema_indicator()

    # --- Momentum ---
    df['rsi_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    #df['atr_14'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    #df['cci_14'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).cci()
    df['stoch_k'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
    df['stoch_d'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch_signal()

    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['macd_momentum'] = df['macd'] - df['macd'].shift(1)

    # --- Volatility ---
    boll = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_bbm'] = boll.bollinger_mavg()
    df['bb_bbh'] = boll.bollinger_hband()
    df['bb_bbl'] = boll.bollinger_lband()

    df['atr'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()

    # --- Volume/Price ratio ---
    df['volume_price_ratio'] = df['Volume'] / df['Close']

    # --- Price momentum ---
    df['price_change_1h'] = df['Close'].pct_change(1)
    df['price_change_3h'] = df['Close'].pct_change(3)
    df['price_change_6h'] = df['Close'].pct_change(6)

    # --- Time (cyclical) ---
    df['hour'] = df['Timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    return df

def create_sequence_data(df, seq_len, future_window, buy_threshold, sell_threshold):
    sequences = []
    labels = []
    indicator_cols = [col for col in df.columns if col not in ['Label', 'year', 'month', 'Timestamp']]

    for i in range(seq_len, len(df) - future_window):
        seq_slice = df.iloc[i - seq_len:i][indicator_cols]
        if seq_slice.isnull().values.any():
            continue
        
        current_close = df.iloc[i - 1]['Close']
        future_close = df.iloc[i + future_window]['Close']
        future_return = (future_close - current_close) / current_close

        if future_return > buy_threshold:
            label = 1  # BUY -> 1
        elif future_return < sell_threshold:
            label = 0  # SELL -> 0
        else:
            label = 2

        sequences.append(seq_slice.values)  # 2D array (seq_len x num_features)
        labels.append(label)

    sequences = np.array(sequences)  # shape: (num_samples, seq_len, num_features)
    labels = np.array(labels)

    return sequences, labels, indicator_cols

def oversample_data(sequences, labels):
    unique_classes, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()

    oversampled_sequences = []
    oversampled_labels = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        cls_seqs = sequences[cls_indices]
        cls_lbls = labels[cls_indices]

        # Liczba powtórzeń (pełnych i częściowych)
        reps = max_count // len(cls_indices)
        remainder = max_count % len(cls_indices)

        oversampled_seqs = np.concatenate([cls_seqs] * reps + [cls_seqs[:remainder]], axis=0)
        oversampled_lbls = np.concatenate([cls_lbls] * reps + [cls_lbls[:remainder]], axis=0)

        oversampled_sequences.append(oversampled_seqs)
        oversampled_labels.append(oversampled_lbls)

    sequences_balanced = np.concatenate(oversampled_sequences, axis=0)
    labels_balanced = np.concatenate(oversampled_labels, axis=0)

    # Przemieszaj dane
    perm = np.random.permutation(len(labels_balanced))
    return sequences_balanced[perm], labels_balanced[perm]

def prepare_features(symbol='BTCUSDT', timeframe=TF, limit=LIMIT, seq_len=SEQ_LEN, future_window=FUTURE_WINDOW,
                    buy_threshold_rest=BUY_THRESHOLD_REST, sell_threshold_rest=SELL_THRESHOLD_REST,
                    buy_threshold_2024_2025=BUY_THRESHOLD_2024_2025, sell_threshold_2024_2025=SELL_THRESHOLD_2024_2025):
    df = fetch_binance_ohlcv(symbol, timeframe, limit)
    df = add_indicators(df)
    df.dropna(inplace=True)

    indicator_cols = [col for col in df.columns if col not in ['Timestamp', 'year', 'month', 'Label']]

    scaler = StandardScaler()
    df[indicator_cols] = scaler.fit_transform(df[indicator_cols])

    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    mask_2024_2025 = ((df['year'] == 2024) & (df['month'] >= 10)) | (df['year'] == 2025)
    df_2024_2025 = df[mask_2024_2025].copy()
    df_rest = df[~mask_2024_2025].copy()

    seq_2024_2025, labels_2024_2025, feature_cols = create_sequence_data(df_2024_2025, seq_len, future_window, buy_threshold_2024_2025, sell_threshold_2024_2025)
    seq_rest, labels_rest, _ = create_sequence_data(df_rest, seq_len, future_window, buy_threshold_rest, sell_threshold_rest)

    seq_all = np.concatenate([seq_rest, seq_2024_2025], axis=0)
    labels_all = np.concatenate([labels_rest, labels_2024_2025], axis=0)

    seq_balanced, labels_balanced = oversample_data(seq_all, labels_all)

    print('Kształt sekwencji po oversamplingu:', seq_balanced.shape)
    print('Etykiety po oversamplingu:', np.bincount(labels_balanced))

    # Dodaj kolumnę NoneNormalized tylko do zapisu CSV
    df['NoneNormalized'] = df['Close'] * scaler.scale_[indicator_cols.index('Close')] + scaler.mean_[indicator_cols.index('Close')]
    cols = ['NoneNormalized'] + [col for col in df.columns if col != 'NoneNormalized']
    df = df[cols]

    import os
    base_dir = 'datasets'
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith('model_') and os.path.isdir(os.path.join(base_dir, d))]
    nums = []
    for d in existing:
        try:
            num = int(d.replace('model_',''))
            nums.append(num)
        except ValueError:
            pass
    next_num = 1 if not nums else max(nums)+1
    model_dir = os.path.join(base_dir, f'model_{next_num}')
    os.makedirs(model_dir, exist_ok=True)

    # Zapisz do unikalnego folderu
    np.save(os.path.join(model_dir, 'btc_sequences.npy'), seq_balanced)
    np.save(os.path.join(model_dir, 'btc_labels.npy'), labels_balanced)
    df.to_csv(os.path.join(model_dir, "btc_raw_data.csv"), index=False)
    print(f"Zapisano dane do folderu: {model_dir}")
    print("Przykładowe cechy:", feature_cols)
    print("Pierwsze 10 etykiet:", labels_balanced[:10])



