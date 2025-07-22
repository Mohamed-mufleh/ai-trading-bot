import requests
import websocket
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
import os


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
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df['hour'] = df['Timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek'] = df['Timestamp'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month

    return df

ohlcv_sequence = []

# Funkcja do pobrania i wyświetlenia początkowych danych
def fetch_initial_ohlcv_and_fit_scaler():
    print("⏳ Pobieram ostatnie 64 świeczki 1h z Binance...")
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': 64
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    candles = []
    for k in data:
        candle = [
            int(k[0]),     # timestamp
            float(k[1]),   # open
            float(k[2]),   # high
            float(k[3]),   # low
            float(k[4]),   # close
            float(k[5]),   # volume
            float(k[9]),   # taker buy base asset volume
            float(k[10])   # taker buy quote asset volume
        ]
        candles.append(candle)

    df = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Taker buy base asset volume', 'Taker buy quote asset volume'])
    df = add_indicators(df)
    df.dropna(inplace=True)
    # Fit scaler na wszystkich historycznych świecach (tylko feature_cols)
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)
    # Zwróć ostatnie 24 świeczki (po dropna) jako sekwencję
    ohlcv_seq = df.tail(24)
    print("\n📊 Ostatnia świeczka z wyliczonymi wskaźnikami:")
    print(ohlcv_seq[['Timestamp', 'Close', 'ema_trend', 'rsi_14', 'macd', 'macd_momentum']])
    print("✅ Otrzymano 24 świeczki. Wyświetlam:")
    print_candles(ohlcv_seq[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].values.tolist())
    # Dodaj porównanie czasu
    last_ts = ohlcv_seq['Timestamp'].iloc[-1]
    print(f"\n🕒 Timestamp ostatniej świecy: {last_ts} (UTC)")
    print(f"🕒 Aktualny czas UTC: {datetime.utcnow()}")

    # --- 🧠 Predykcja dla ostatniej świecy z historii ---
    input_seq = ohlcv_seq[feature_cols].values.astype(np.float32)
    input_seq = scaler.transform(input_seq)
    input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)  # (1, 24, features)
    with torch.no_grad():
        logits = model(input_tensor)
        pred_class = torch.argmax(logits, dim=1).item()
    label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    prediction = label_map[pred_class]
    last_ts = ohlcv_seq['Timestamp'].iloc[-1]
    print(f"\n🚨 MODEL PREDYKCJA dla świecy z {last_ts}: >>> {prediction} <<<")
    # --- Składanie zlecenia na podstawie predykcji ---
    place_order(prediction)

    # --- 🧠 Predykcja dla przedostatniej świecy z historii ---
    if len(ohlcv_seq) >= 2:
        input_seq_prev = ohlcv_seq.iloc[:-1][feature_cols].values.astype(np.float32)
        input_seq_prev = scaler.transform(input_seq_prev)
        input_tensor_prev = torch.tensor(input_seq_prev).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_prev = model(input_tensor_prev)
            pred_class_prev = torch.argmax(logits_prev, dim=1).item()
        prediction_prev = label_map[pred_class_prev]
        prev_ts = ohlcv_seq['Timestamp'].iloc[-2]
        print(f"\n🚨 MODEL PREDYKCJA dla świecy z {prev_ts}: >>> {prediction_prev} <<<")

    # --- 🧠 Predykcja dla trzeciej od końca świecy z historii ---
    if len(ohlcv_seq) >= 3:
        input_seq_prev2 = ohlcv_seq.iloc[:-2][feature_cols].values.astype(np.float32)
        input_seq_prev2 = scaler.transform(input_seq_prev2)
        input_tensor_prev2 = torch.tensor(input_seq_prev2).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_prev2 = model(input_tensor_prev2)
            pred_class_prev2 = torch.argmax(logits_prev2, dim=1).item()
        prediction_prev2 = label_map[pred_class_prev2]
        prev2_ts = ohlcv_seq['Timestamp'].iloc[-3]
        print(f"\n🚨 MODEL PREDYKCJA dla świecy z {prev2_ts}: >>> {prediction_prev2} <<<")

    return ohlcv_seq, scaler

# Funkcja pomocnicza do wypisania świeczek
def print_candles(candles):
    print(f"\n{'#':>2} | {'Time':<16} | {'Open':>10} | {'High':>10} | {'Low':>10} | {'Close':>10} | {'Vol':>10}")
    print("-" * 80)
    for i, c in enumerate(candles):
        # Obsługa Timestamp lub int (ms)
        if isinstance(c[0], pd.Timestamp):
            t = c[0].strftime('%Y-%m-%d %H:%M')
        else:
            t = datetime.fromtimestamp(c[0] / 1000).strftime('%Y-%m-%d %H:%M')
        print(f"{i+1:>2} | {t:<16} | {c[1]:>10.2f} | {c[2]:>10.2f} | {c[3]:>10.2f} | {c[4]:>10.2f} | {c[5]:>10.2f}")
    print("-" * 80)


# WebSocket: odbiór świeczek i aktualizacja sekwencji
def on_message(ws, message):
    global ohlcv_sequence, scaler

    data = json.loads(message)
    kline = data.get('k', {})

    if kline.get('x'):  # zakończona świeca
        new_candle = [
            int(kline['t']),
            float(kline['o']),
            float(kline['h']),
            float(kline['l']),
            float(kline['c']),
            float(kline['v']),
            float(kline['V']),  # taker buy base asset volume
            float(kline['Q'])   # taker buy quote asset volume
        ]

        # Dodaj nową świecę do sekwencji
        ohlcv_sequence = ohlcv_sequence.append(pd.DataFrame([new_candle], columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Taker buy base asset volume', 'Taker buy quote asset volume']), ignore_index=True)
        # Wylicz wskaźniki na całej sekwencji
        df = add_indicators(ohlcv_sequence)
        df.dropna(inplace=True)
        # Zostaw tylko ostatnie 24 świeczki
        df = df.tail(24)
        ohlcv_sequence = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Taker buy base asset volume', 'Taker buy quote asset volume']].copy()

        print("\n🕒 Nowa świeczka 1h zakończona. Aktualna sekwencja 24 OHLCV:")
        print_candles(ohlcv_sequence.values.tolist())

        # --- 🧠 Predykcja ---
        input_seq = df[feature_cols].values.astype(np.float32)
        input_seq = scaler.transform(input_seq)
        input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)  # (1, 24, features)

        with torch.no_grad():
            logits = model(input_tensor)
            pred_class = torch.argmax(logits, dim=1).item()

        label_map = {0: "SELL", 2: "HOLD", 1: "BUY"}
        prediction = label_map[pred_class]
        last_timestamp = df.iloc[-1]['Timestamp']
        print(f"\n🚨 MODEL PREDYKCJA dla świecy z {last_timestamp}: >>> {prediction} <<<")


# Uruchomienie WebSocketu
def start_websocket():
    ws_url = 'wss://stream.binance.com:9443/ws/btcusdt@kline_1h'
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    ws.run_forever()

# Lista cech wejściowych używanych przez model (identyczna jak w newe_data.py, bez Timestamp, year, month, Label)
feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Taker buy base asset volume', 'Taker buy quote asset volume',
    'ema_9', 'ema_21', 'ema_trend', 'rsi_14', 'stoch_k', 'stoch_d',
    'macd', 'macd_signal', 'macd_diff', 'macd_momentum',
    'bb_bbm', 'bb_bbh', 'bb_bbl', 'atr',
    'volume_price_ratio',
    'price_change_1h', 'price_change_3h', 'price_change_6h',
    'hour', 'hour_sin', 'hour_cos', 'dayofweek', 'dow_sin', 'dow_cos'
]

INPUT_FEATURES = len(feature_cols)

class CNN_GRU_Model(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=3):
        super(CNN_GRU_Model, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)

        self.relu = nn.ReLU()
        self.gru = nn.GRU(32, hidden_size, batch_first=True,  num_layers=2, bidirectional=True)

        self.dropout_fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.permute(0, 2, 1)  # (B, T, C) for GRU
        _, h = self.gru(x)
        h_forward = h[-2, :, :]  # ostatnia warstwa, forward
        h_backward = h[-1, :, :]  # ostatnia warstwa, backward
        h_combined = torch.cat((h_forward, h_backward), dim=1)  # (B, 2*hidden_size)

        x = self.dropout_fc(h_combined)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- KONFIGURACJA BINANCE API ---
API_KEY = ''
API_SECRET = ''
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

# --- GLOBALNY BALANS POCZĄTKOWY ---
initial_balance = None
balance_log_path = 'orders_log.csv'

def get_and_save_initial_balance():
    global initial_balance
    try:
        balance = client.get_asset_balance(asset='USDT')
        initial_balance = float(balance['free'])
        # Zapisz do pliku CSV jeśli nie istnieje
        if not os.path.exists(balance_log_path):
            with open(balance_log_path, 'w') as f:
                f.write('timestamp,order_type,qty,price,usdt_before,usdt_after,usdt_diff\n')
        print(f"💰 Początkowy balans USDT: {initial_balance}")
    except Exception as e:
        print(f"❌ Błąd przy pobieraniu początkowego balansu: {e}")
        initial_balance = 0.0

# --- Funkcja do składania zleceń ---
def place_order(prediction, symbol='BTCUSDT', percent=0.01):
    global initial_balance
    try:
        # Pobierz balans USDT
        balance = client.get_asset_balance(asset='USDT')
        usdt_balance = float(balance['free'])
        # Pobierz aktualną cenę BTCUSDT
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        # Pobierz info o symbolu (filtry)
        info = client.get_symbol_info(symbol)
        lot_size = None
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                min_qty = float(f['minQty'])
                step_size = float(f['stepSize'])
                break
        else:
            min_qty = 0.00001
            step_size = 0.00001
        if prediction == 'SELL':
            # Sprzedaj 50% posiadanych BTC
            btc_balance = client.get_asset_balance(asset='BTC')
            btc_qty = float(btc_balance['free'])
            qty = btc_qty #T* 0.5
            precision = int(round(-np.log10(step_size)))
            qty = round(qty, precision)
            if qty < min_qty:
                print(f"❌ Wyliczona ilość {qty} < minQty {min_qty} dla {symbol}. Zlecenie nie zostanie złożone.")
                return
            usdt_to_use = qty * price
        else:
            # Kup za percent USDT
            usdt_to_use = usdt_balance * percent
            qty = usdt_to_use / price
            precision = int(round(-np.log10(step_size)))
            qty = round(qty, precision)
            if qty < min_qty:
                print(f"❌ Wyliczona ilość {qty} < minQty {min_qty} dla {symbol}. Zlecenie nie zostanie złożone.")
                return
    except Exception as e:
        print(f"❌ Błąd przy pobieraniu balansu, ceny lub filtrów: {e}")
        return

    usdt_before = usdt_balance
    order_type = None
    order_id = None
    if prediction == 'BUY':
        order_type = 'BUY'
        print(f"\n🟢 Składam zlecenie KUPNA {qty} {symbol} (za {usdt_to_use:.2f} USDT)...")
        try:
            order = client.create_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=qty
            )
            order_id = order['orderId']
            print(f"✅ Zlecenie KUPNA złożone: {order_id}")
        except Exception as e:
            print(f"❌ Błąd przy składaniu zlecenia KUPNA: {e}")
    elif prediction == 'SELL':
        order_type = 'SELL'
        print(f"\n🔴 Składam zlecenie SPRZEDAŻY {qty} {symbol} (50% posiadanych BTC)...")
        try:
            order = client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=qty
            )
            order_id = order['orderId']
            print(f"✅ Zlecenie SPRZEDAŻY złożone: {order_id}")
        except Exception as e:
            print(f"❌ Błąd przy składaniu zlecenia SPRZEDAŻY: {e}")
    else:
        print("🟡 HOLD – nie składam zlecenia, czekam na kolejną świecę.")
        return

    # Po zleceniu pobierz nowy balans i zapisz do CSV
    try:
        balance_after = client.get_asset_balance(asset='USDT')
        usdt_after = float(balance_after['free'])
        usdt_diff = usdt_after - usdt_before
        with open(balance_log_path, 'a') as f:
            f.write(f"{datetime.utcnow()},{order_type},{qty},{price},{usdt_before},{usdt_after},{usdt_diff}\n")
        print(f"💾 Zapisano zlecenie do {balance_log_path}. Różnica balansu: {usdt_diff:.2f} USDT")
    except Exception as e:
        print(f"❌ Błąd przy zapisie zlecenia do pliku: {e}")

# Parametry
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_GRU_Model(input_size=INPUT_FEATURES, num_classes=3)
model.load_state_dict(torch.load('model_gru_cnn_100.pth', map_location=device))
model.to(device)
model.eval()

# Główna funkcja
if __name__ == "__main__":
    print("Bot uruchomiony w trybie godzinowym. Każda pełna godzina +1 min pobiera dane i wyświetla predykcję.")
    get_and_save_initial_balance()
    fetch_initial_ohlcv_and_fit_scaler()
    try:
        while True:
            now = datetime.utcnow()
            # Oblicz czas do następnej pełnej godziny + 1 min
            next_hour = (now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1, minutes=1))
            wait_seconds = (next_hour - now).total_seconds()
            print(f"Czekam {int(wait_seconds)} sekund do kolejnej predykcji o {next_hour.strftime('%H:%M')} UTC...")
            time.sleep(wait_seconds)
            # Pobierz i przetwórz świeczki, wyświetl predykcję
            fetch_initial_ohlcv_and_fit_scaler()
    except KeyboardInterrupt:
        print("\nZatrzymano program.")
