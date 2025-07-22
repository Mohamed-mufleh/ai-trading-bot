import torch
import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
import os

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

def run_backtest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Wczytaj dane z najnowszego folderu datasets/model_N ---
    datasets_dir = 'datasets'
    model_dirs = [d for d in os.listdir(datasets_dir) if d.startswith('model_') and os.path.isdir(os.path.join(datasets_dir, d))]
    nums = []
    for d in model_dirs:
        try:
            num = int(d.replace('model_',''))
            nums.append(num)
        except ValueError:
            pass
    if not nums:
        raise FileNotFoundError('Brak folderów model_N w datasets')
    latest_num = max(nums)
    model_dir = os.path.join(datasets_dir, f'model_{latest_num}')
    print(f"Wczytuję dane z folderu: {model_dir}")
    X = np.load(os.path.join(model_dir, 'btc_sequences.npy'))
    y = np.load(os.path.join(model_dir, 'btc_labels.npy'))
    import pandas as pd
    df = pd.read_csv(os.path.join(model_dir, 'btc_raw_data.csv'))

    # Konwersja do torch
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    # Znajdź najwyższy numer pliku model_gru_cnn_N.pth w folderze models
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith('model_gru_cnn_') and f.endswith('.pth')]
    nums = []
    for f in model_files:
        try:
            num = int(f.replace('model_gru_cnn_','').replace('.pth',''))
            nums.append(num)
        except ValueError:
            pass
    if not nums:
        raise FileNotFoundError('Brak plików model_gru_cnn_N.pth w folderze models')
    latest_num = max(nums)
    model_path = os.path.join(models_dir, f'model_gru_cnn_{latest_num}.pth')
    print(f"Ładuję model: {model_path}")

    model = CNN_GRU_Model(input_size=X.shape[2], num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    batch_size = 64  # możesz zmienić na mniejszy jeśli pamięć dalej za mała

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())

    preds = torch.cat(all_preds).numpy()  # indeks klasy
    y_true = y_tensor.cpu().numpy()

    # Raport i metryki
    print(classification_report(y_true, preds, target_names=['SELL', 'HOLD', 'BUY']))
    print("Confusion matrix:\n", confusion_matrix(y_true, preds))
    print("F1 score (macro):", f1_score(y_true, preds, average='macro'))

    SEQ_LEN = 24

    def simulate_strategy_precise(preds, true_labels, df, seq_len):
        """
        Bardzo dokładna symulacja strategii: zawsze mamy albo USDT, albo BTC.
        Brak prowizji. Equity liczone w USDT po każdej transakcji.
        Używa ceny z kolumny NoneNormalized.
        """
        start_idx = seq_len
        close_prices = df['NoneNormalized'].values
        results = []
        usdt = 1.0  # zaczynamy z 1 USDT
        btc = 0.0
        position = 'USDT'  # 'USDT' lub 'BTC'
        entry_price = None
        entry_idx = None
        for i, (pred, true_label) in enumerate(zip(preds, true_labels)):
            idx = start_idx + i
            if idx >= len(close_prices):
                break
            price = close_prices[idx]
            if pred == 1:
                continue  # HOLD
            # BUY
            if pred == 2 and position == 'USDT':
                btc = usdt / price
                usdt = 0.0
                position = 'BTC'
                entry_price = price
                entry_idx = idx
                results.append({
                    'Timestamp': df.iloc[idx]['Timestamp'],
                    'Action': 'BUY',
                    'Price': price,
                    'USDT': usdt,
                    'BTC': btc,
                    'Equity': btc * price,
                    'True_Label': ['SELL','HOLD','BUY'][true_label],
                    'Pred': 'BUY'
                })
            # SELL
            elif pred == 0 and position == 'BTC':
                usdt = btc * price
                btc = 0.0
                position = 'USDT'
                results.append({
                    'Timestamp': df.iloc[idx]['Timestamp'],
                    'Action': 'SELL',
                    'Price': price,
                    'USDT': usdt,
                    'BTC': btc,
                    'Equity': usdt,
                    'True_Label': ['SELL','HOLD','BUY'][true_label],
                    'Pred': 'SELL'
                })
            # Jeśli pred == 2 i już mamy BTC, albo pred == 0 i już mamy USDT, nic nie robimy
        # Na koniec przelicz equity na USDT jeśli trzymamy BTC
        if btc > 0:
            final_price = close_prices[-1]
            usdt = btc * final_price
            results.append({
                'Timestamp': df.iloc[-1]['Timestamp'],
                'Action': 'FINAL_SELL',
                'Price': final_price,
                'USDT': usdt,
                'BTC': 0.0,
                'Equity': usdt,
                'True_Label': 'N/A',
                'Pred': 'SELL'
            })
        return pd.DataFrame(results)

    # --- Użycie bardzo dokładnej symulacji ---
    strategy_df = simulate_strategy_precise(preds, y_true, df, SEQ_LEN)
    strategy_df.to_csv("strategy_results_precise.csv", index=False)

    print("\n--- Bardzo dokładna symulacja strategii (bez prowizji, cena z NoneNormalized) ---")
    print(strategy_df[['Timestamp','Action','Price','USDT','BTC','Equity','True_Label','Pred']])

    final_equity = strategy_df['Equity'].iloc[-1]
    profit_pct = (final_equity - 1.0) * 100
    print(f"\nZysk końcowy: {profit_pct:.2f}% (start: 1 USDT, koniec: {final_equity:.4f} USDT)")
    print(f"Liczba transakcji: {len(strategy_df)}")

    # --- Zapisz wynik do results_summary.csv ---
    results_path = "results_summary.csv"
    model_name = os.path.basename(model_path)
    row = {"Model": model_name, "Score": profit_pct}
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
    else:
        df_results = pd.DataFrame([row])
    df_results.to_csv(results_path, index=False)
