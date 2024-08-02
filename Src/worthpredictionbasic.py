import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import joblib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

batch_size = 32
num_epochs = 300
load_model = False

data = pd.read_csv('./Data/worth_data.csv')
data = data.drop('index',axis = 1)
X = data.drop('Value',axis = 1)
y = data['Value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset =test_dataset, batch_size=batch_size, shuffle=False)

class PlayerWorthPredictor(nn.Module):
    def __init__(self):
        super(PlayerWorthPredictor, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = PlayerWorthPredictor()
model = model.to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if load_model == True:
    model = joblib.load('./model/baseworthpred.pkl')

summary(model, (37,))

record = {
    "Train Loss": [],
    "Val Loss": [],
    "Train MSE": [],
    "Val MSE": [],
}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_predictions = []
    train_targets = []

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_predictions.extend(outputs.cpu().detach().numpy())
        train_targets.extend(targets.cpu().detach().numpy())

    train_loss /= len(train_loader.dataset)
    train_mse = mean_squared_error(train_targets, train_predictions)

    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            val_predictions.extend(outputs.cpu().detach().numpy())
            val_targets.extend(targets.cpu().detach().numpy())

    val_loss /= len(val_loader.dataset)
    val_mse = mean_squared_error(val_targets, val_predictions)

    record["Train Loss"].append(train_loss)
    record["Val Loss"].append(val_loss)
    record["Train MSE"].append(train_mse)
    record["Val MSE"].append(val_mse)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    print(f"Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
    # Calculate R2 score for training set
    train_r2 = r2_score(y_train, train_predictions)

    # Calculate R2 score for validation set
    val_r2 = r2_score(val_targets, val_predictions)

    print(f"Training R2 Score: {train_r2:.4f}")
    print(f"Validation R2 Score: {val_r2:.4f}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

plt.plot(record["Train Loss"], label="Train")
plt.plot(record["Val Loss"], label = "Val")
plt.legend()
plt.plot()
plt.show()
# joblib.dump(model, 'baseworthpred.pkl')
# joblib.dump(scaler, 'baseworthscaler.pkl')
torch.save(model.state_dict(), './model/worth_model_file.pth')
