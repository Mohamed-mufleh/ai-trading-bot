import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

def train_model(batch_size=64, epochs=55):
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

    print(f"Kształt X: {X.shape}, kształt y: {y.shape}")
    print(f"Przykładowe etykiety: {y[:10]}")

    # Podział danych na train (70%), val (15%), test (15%)
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Dataset i DataLoader
    class SequenceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN_GRU_Model(input_size=X.shape[2]).to(DEVICE)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
    print("Class weights:", class_weights)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).long()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).long()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

    # --- ZAPIS MODELU Z UNIKALNYM NUMEREM DO FOLDERU models ---
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    existing = [f for f in os.listdir(models_dir) if f.startswith('model_gru_cnn_') and f.endswith('.pth')]
    nums = []
    for f in existing:
        try:
            num = int(f.replace('model_gru_cnn_','').replace('.pth',''))
            nums.append(num)
        except ValueError:
            pass
    next_num = 1 if not nums else max(nums)+1
    model_path = os.path.join(models_dir, f'model_gru_cnn_{next_num}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model zapisany do: {model_path}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    print(classification_report(y_true, y_pred, target_names=["SELL", "HOLD", "BUY"]))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score (macro): {f1:.4f}")




