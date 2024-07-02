import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the Excel file
file_path = r'C:\Users\tudor\OneDrive\Desktop\facultate\Licenta\antrenare model\weatherdata1hr2019_2024.xlsx'
data = pd.read_excel(file_path)

# Combine the date and time into a single datetime column
data['datetime'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
data.set_index('datetime', inplace=True)

# Drop the original date and time columns
data.drop(columns=['Year', 'Month', 'Day', 'time', 'date'], inplace=True)

# Feature Engineering: Add time-based features
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        labels.append(data[i+sequence_length])
    return np.array(sequences), np.array(labels)

sequence_length = 60  # Using 60 time steps for each sequence
X, y = create_sequences(scaled_data, sequence_length)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the LSTM model with increased complexity
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = data.shape[1]
hidden_size = 256  # Increased hidden size
output_size = data.shape[1]
num_layers = 4
dropout = 0.3  # Increased dropout rate
num_epochs = 100  # Increased epochs
learning_rate = 0.0001  # Lowered learning rate

# Instantiate the model, define the loss function and the optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model with early stopping
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Early stopping
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    print(f'Validation Loss: {val_loss:.4f}')
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_weather_prediction_lstm.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping due to no improvement in validation loss.")
            break

# Load the best model for prediction
model.load_state_dict(torch.load('best_weather_prediction_lstm.pth'))

# Select the most recent sequence from the dataset for prediction
recent_sequence = scaled_data[-sequence_length:]
recent_sequence = np.expand_dims(recent_sequence, axis=0)  # Add batch dimension
recent_sequence_tensor = torch.tensor(recent_sequence, dtype=torch.float32).to(device)

# Predict future data points for the next week (7 days, 24 hours each)
future_steps = 7 * 24  # Predict the next 7 days
future_predictions = []

with torch.no_grad():
    for _ in range(future_steps):
        # Make the prediction
        predicted = model(recent_sequence_tensor)
        future_predictions.append(predicted.cpu().numpy())
        
        # Update the recent sequence with the new prediction
        predicted = np.expand_dims(predicted.cpu().numpy(), axis=0)
        new_sequence = np.append(recent_sequence_tensor.cpu().numpy(), predicted, axis=1)
        recent_sequence_tensor = torch.tensor(new_sequence[:, -sequence_length:, :], dtype=torch.float32).to(device)

# Convert predictions to numpy array and reshape
future_predictions = np.array(future_predictions).reshape(-1, output_size)

# Inverse transform to get actual values
future_predictions_actual = scaler.inverse_transform(future_predictions)

# Create a DataFrame to store the results
future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='H')[1:]
future_df = pd.DataFrame(future_predictions_actual, index=future_dates, columns=data.columns)

# Save the predictions to a CSV file
future_df.to_csv('predicted_weather_next_week.csv')

print(future_df)
