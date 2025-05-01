import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Custom Dataset for HSI data
class HSIDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ViT-inspired model for 1D spectral data
class HSIViT(nn.Module):
    def __init__(self, num_classes, num_bands=256, patch_size=16, hidden_size=384, num_layers=6, num_heads=6, dropout=0.1):
        super(HSIViT, self).__init__()
        self.num_patches = num_bands // patch_size
        self.hidden_size = hidden_size

        # Patch embedding layer
        self.patch_embedding = nn.Linear(patch_size, hidden_size)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, 256)
        batch_size = x.size(0)
        
        # Split into patches: (batch_size, num_patches, patch_size)
        x = x.view(batch_size, self.num_patches, -1)  # (batch_size, 16, 16)
        
        # Project patches to hidden_size
        x = self.patch_embedding(x)  # (batch_size, num_patches, hidden_size)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, hidden_size)
        
        # Add positional encoding
        x = x + self.positional_encoding
        x = self.dropout(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, num_patches + 1, hidden_size)
        
        # Extract CLS token output
        cls_output = x[:, 0, :]  # (batch_size, hidden_size)
        
        # Classify
        logits = self.classifier(cls_output)  # (batch_size, num_classes)
        return logits

# Load and preprocess data
def load_and_preprocess_data(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Extract features (band_1 to band_256) and labels (grain_type)
    feature_cols = [f'band_{i+1}' for i in range(256)]
    X = df[feature_cols].values  # Shape: (n_samples, 256)
    #y = df['grain_type'].values  # Shape: (n_samples,)
    #dflabel = df['grain_type'].apply(lambda x:Path(x).stem.split("-")[0])
    dflabel1 = df['grain_type'].str.extract(r"([^/]+)(?=-\d+\.hdr$)", expand=False)
    print(X.shape)
    print(dflabel1.shape)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(dflabel1)
    print(len(y_encoded))
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #print(X_scaled)

    # Split data: 80% train, 10% validation, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    print(X_train)
    print(X_temp)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}')

# Evaluation function
def evaluate_model(model, test_loader, device, label_encoder):
    model.eval()
    test_correct = 0
    test_total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Decode predictions to original labels
    predictions_decoded = label_encoder.inverse_transform(predictions)
    true_labels_decoded = label_encoder.inverse_transform(true_labels)
    
    return predictions_decoded, true_labels_decoded

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess data
    csv_path = 'full_grain_data.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_and_preprocess_data(csv_path)
    
    # Create datasets and dataloaders
    train_dataset = HSIDataset(X_train, y_train)
    val_dataset = HSIDataset(X_val, y_val)
    test_dataset = HSIDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    num_classes = len(label_encoder.classes_)
    model = HSIViT(num_classes=num_classes).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device, num_epochs=20)
    
    # Evaluate model
    predictions, true_labels = evaluate_model(model, test_loader, device, label_encoder)
    
    # Save model
    torch.save(model.state_dict(), 'hsivit_model.pth')
    print("Model saved to 'hsivit_model.pth'")
    

    # Save predictions to CSV
    results = pd.DataFrame({'True': true_labels, 'Predicted': predictions})
    results.to_csv('predictions.csv', index=False)
    # Print some example predictions
    print("\nExample Predictions:")
    for pred, true in zip(predictions[:5], true_labels[:5]):
        print(f"Predicted: {pred}, True: {true}")