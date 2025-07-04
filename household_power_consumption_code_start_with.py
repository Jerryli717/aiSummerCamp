# %%
import numpy as np
import pandas as pd

# %%
# load data
df = pd.read_csv(r'D:\lzy\华科\课件\招商证券工程训练营\aiSummerCamp2025-master\aiSummerCamp2025-master\day3\assignment\data\household_power_consumption\household_power_consumption.txt', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']
# data normalization

# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler
feature_cols = [col for col in train.columns if col not in ['datetime', 'Global_active_power']]
scaler = MinMaxScaler()
train_scaled = train.copy()
test_scaled = test.copy()
train_scaled[feature_cols + ['Global_active_power']] = scaler.fit_transform(train[feature_cols + ['Global_active_power']])
test_scaled[feature_cols + ['Global_active_power']] = scaler.transform(test[feature_cols + ['Global_active_power']])

# %%
# split X and y
import numpy as np
def create_sequences(data, seq_length=24):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)][feature_cols].values
        y = data.iloc[i+seq_length]['Global_active_power']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 24
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# %%
# creat dataloaders
import torch
from torch.utils.data import TensorDataset, DataLoader
batch_size = 64
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# build a LSTM model
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=len(feature_cols)).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.6f}")

# %%
# evaluate the model on the test set
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        trues.append(yb.numpy())
preds = np.concatenate(preds)
trues = np.concatenate(trues)

# 反归一化
y_scaler = MinMaxScaler()
y_scaler.fit(train[['Global_active_power']])
preds_inv = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
trues_inv = y_scaler.inverse_transform(trues.reshape(-1, 1)).flatten()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(trues_inv, preds_inv)
print(f"Test MSE: {mse:.4f}")

# %%
# plotting the predictions against the ground truth
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.plot(trues_inv[:500], label='True')
plt.plot(preds_inv[:500], label='Predicted')
plt.legend()
plt.title('LSTM Prediction vs Ground Truth')
plt.xlabel('Time Step')
plt.ylabel('Global_active_power')
plt.show()

# %%
# data normalization

# %%
# split X and y

# %%
# creat dataloaders

# %%
# build a LSTM model

# %%
# train the model

# %%
# evaluate the model on the test set

# %%
# plotting the predictions against the ground truth
