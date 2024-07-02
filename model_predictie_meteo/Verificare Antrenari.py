import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the Excel file
file_path = r'C:\Users\tudor\OneDrive\Desktop\facultate\Licenta\antrenare model\weatherdata1hr2019_2024.xlsx'
data = pd.read_excel(file_path)

# Combine date and time into a single datetime column and set as index
data['datetime'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
data.set_index('datetime', inplace=True)
data.drop(columns=['Year', 'Month', 'Day', 'time', 'date'], inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length])
    return np.array(sequences), np.array(labels)

# Preparing data
sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)

# Split data into training and testing
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters and device setup
input_size = data.shape[1]
hidden_size = 200
output_size = data.shape[1]
num_epochs = 20
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training and evaluation functions
results = {'num_layers': [], 'MAE': [], 'MSE': [], 'RMSE': []}

def train_and_evaluate(num_layers):
    model = LSTMModel(input_size, hidden_size, output_size, num_layers, device).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training with {num_layers} LSTM layers...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] Complete - Loss: {avg_loss:.4f}')

    model.eval()
    test_predictions, test_targets = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    test_predictions = scaler.inverse_transform(test_predictions)
    test_targets = scaler.inverse_transform(test_targets)

    mae = mean_absolute_error(test_targets, test_predictions)
    mse = mean_squared_error(test_targets, test_predictions)
    rmse = np.sqrt(mse)
    results['num_layers'].append(num_layers)
    results['MAE'].append(mae)
    results['MSE'].append(mse)
    results['RMSE'].append(rmse)
    print(f'Num Layers: {num_layers}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

for num_layers in range(4, 9):
    train_and_evaluate(num_layers)

# Plotting the results
plt.figure(figsize=(10, 8))
plt.plot(results['num_layers'], results['MSE'], label='MSE')
plt.plot(results['num_layers'], results['RMSE'], label='RMSE')
plt.plot(results['num_layers'], results['MAE'], label='MAE')
plt.xlabel('Number of Layers')
plt.ylabel('Error')
plt.title('Model Error vs. Number of LSTM Layers')
plt.legend()
plt.grid(True)
plt.show()
