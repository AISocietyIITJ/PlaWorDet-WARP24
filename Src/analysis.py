import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
from tqdm import tqdm
from torchsummary import summary
from sympy import false
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

worth_data = pd.read_csv('../Data/worth_data.csv')
worth_data = worth_data.drop('index',axis = 1)
X_worth = worth_data.drop('Value',axis = 1)
y_worth = worth_data['Value']
worth_scaler = StandardScaler()
# worth_scaler = joblib.load('../model/baseworthpred.pkl')
X_worth = worth_scaler.fit_transform(X_worth) ## changes
X_worth_tensor = torch.tensor(X_worth, dtype=torch.float32).to(device)
y_worth_tensor = torch.tensor(y_worth.values, dtype=torch.float32).view(-1, 1).to(device)

## creating position_data
# pos_data = pd.read_csv('../Data/CompleteDatasetmayank.csv', encoding='unicode escape', low_memory=False)

pos_data = pd.read_csv('../Data/worth_data.csv')
input_features = ['Aggression', 'Agility', 'Ball control', 'Curve', 'Dribbling','Finishing'
        ,'FK accuracy', 'Heading accuracy', 'Interceptions',
        'Jumping', 'Long shots', 'Penalties', 'Physical / Positioning',
        'Shot power', 'Strength', 'Vision', 'Volleys',
        'Slide Tack']

pos_data = pos_data[input_features]

pos_scaler = StandardScaler()
# pos_scaler = joblib.load('../model/poscalerfin.pkl')
X_pos = pos_scaler.fit_transform(pos_data) ## changes

X_pos_tensor = torch.tensor(X_pos, dtype=torch.float)

class PlayerWorthPredictor(nn.Module):
    def __init__(self):
        super(PlayerWorthPredictor, self).__init__()
        self.fc1 = nn.Linear(X_worth_tensor.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PositionPredictor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Unified dropout rate

        # Simplified architecture
        self.layer_1 = nn.Linear(in_features, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, out_features)

        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.layer_1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.layer_2(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm3(self.layer_3(x)))
        x = self.dropout(x)
        x = self.layer_4(x)
        output = self.logsoftmax(x)
        return output


model_worth = PlayerWorthPredictor()
model_worth = model_worth.to(device)

model_pos = PositionPredictor(18, 15).to(device)

model_worth.load_state_dict(torch.load('../model/worth_model_file.pth'))
model_pos.load_state_dict(torch.load('../model/pos_model_file.pth'))

# Predict player worth
model_worth.eval()
with torch.no_grad():
    predicted_worth = model_worth(X_worth_tensor).cpu().numpy()

# Predict player position
model_pos.eval()
with torch.no_grad():
    predicted_positions = model_pos(X_pos_tensor).cpu().numpy().argmax(axis=1)

# Create a DataFrame with predictions and actual values
worth_data['Predicted_Worth'] = predicted_worth
worth_data['Predicted_Position'] = predicted_positions

# Group by position and calculate mean worth
actual_worth_by_position = worth_data.groupby('Predicted_Position')['Value'].mean()
predicted_worth_by_position = worth_data.groupby('Predicted_Position')['Predicted_Worth'].mean()

# Define position labels
position_labels = ['ST', 'RW', 'LW', 'CDM', 'CB', 'RM', 'CM', 'LM', 'LB', 'CAM', 'RB', 'CF', 'RWB', 'LWB', 'GK']

# Plot bar graph
positions = actual_worth_by_position.index
actual_worth_values = actual_worth_by_position.values
predicted_worth_values = predicted_worth_by_position.values

fig, ax = plt.subplots(figsize=(12, 6))
width = 0.35  # width of the bars

ax.bar(positions - width/2, actual_worth_values, width, label='Actual Worth')
ax.bar(positions + width/2, predicted_worth_values, width, label='Predicted Worth')

ax.set_xlabel('Player Position')
ax.set_ylabel('Worth')
ax.set_title('Actual vs Predicted Player Worth by Position')
ax.legend()

plt.xticks(positions, [position_labels[i] for i in positions])
plt.show()
