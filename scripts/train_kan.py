from models.kan import KAN
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Extended KAN Model with Correct Batch Normalization
class ExtendedKAN(KAN):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, dropout=0.3):
        super(ExtendedKAN, self).__init__(layers_hidden, grid_size, spline_order)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        in_features = layers_hidden[0]
        
        # Define layers with BatchNorm and Dropout
        for out_features in layers_hidden[1:]:
            self.layers.append(nn.Linear(in_features, out_features))
            self.batch_norms.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = batch_norm(layer(x))
            x = torch.relu(x)  # Activation function
            x = self.dropout(x)
        return x

# Instantiate model
model = ExtendedKAN([4, 64, 32, 1], grid_size=5, spline_order=2, dropout=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and Prepare Dataset
data_path = '/mnt/RAID/projects/PCS956-Project/data/synthetic_vineyard_data.parquet'
data = pd.read_parquet(data_path)

def prepare_data(data, risk_threshold=0.5):
    data['Botrytis_Risk_Class'] = (data['Botrytis_Risk'] > risk_threshold).astype(int)
    filtered_data = data.dropna(subset=['DTM', 'CHM', 'NDVI', 'LAI', 'Botrytis_Risk'])
    X = filtered_data[['DTM', 'CHM', 'NDVI', 'LAI']].values
    y = filtered_data['Botrytis_Risk_Class'].values
    return X, y

X, y = prepare_data(data)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Weighted Sampling for Imbalanced Dataset
class_counts = np.bincount(y_train)
weights = 1.0 / class_counts
sample_weights = weights[y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
trainloader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define loss function with positional weight
pos_weight = torch.tensor([len(y) / np.bincount(y)[1]], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-3)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            accuracy = (torch.sigmoid(output).round() == labels).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels in valloader:
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            val_loss += criterion(output, labels).item()
            predictions = torch.sigmoid(output).round()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            val_accuracy += (predictions == labels).float().mean().item()
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)
    scheduler.step(epoch + len(trainloader))

    # Print Prediction Distribution
    unique, counts = np.unique(all_predictions, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    print(f"Label distribution in validation: {dict(zip(unique_labels, label_counts))}")
    print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
