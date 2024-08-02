import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
split_frac = 0.7

# Load and prepare data
data = pd.read_csv('./Data/worth_data.csv')
data = data.sample(frac=1).reset_index(drop=True)

worth_data = data.drop('index', axis=1)
worth_data_train = worth_data[:int(split_frac * len(worth_data))].copy()
worth_data_test = worth_data[int(split_frac * len(worth_data)):].copy()

X_train_worth = worth_data_train.drop('Value', axis=1)
X_test_worth = worth_data_test.drop('Value', axis=1)
y_train_worth = worth_data_train['Value']
y_test_worth = worth_data_test['Value']

worth_scaler = StandardScaler()
X_train_worth = worth_scaler.fit_transform(X_train_worth)
X_test_worth = worth_scaler.transform(X_test_worth)
X_train_worth_tensor = torch.tensor(X_train_worth, dtype=torch.float32).to(device)
X_test_worth_tensor = torch.tensor(X_test_worth, dtype=torch.float32).to(device)
y_train_worth_tensor = torch.tensor(y_train_worth.values, dtype=torch.float32).view(-1, 1).to(device)
y_test_worth_tensor = torch.tensor(y_test_worth.values, dtype=torch.float32).view(-1, 1).to(device)

# Creating position_data
pos_data = data

input_features = ['Aggression', 'Agility', 'Ball control', 'Curve', 'Dribbling', 'Finishing',
                  'FK accuracy', 'Heading accuracy', 'Interceptions',
                  'Jumping', 'Long shots', 'Penalties', 'Physical / Positioning',
                  'Shot power', 'Strength', 'Vision', 'Volleys',
                  'Slide Tack']

pos_data = pos_data[input_features]
X_train_pos = pos_data[:int(split_frac * len(pos_data))].copy()
X_test_pos = pos_data[int(split_frac * len(pos_data)):].copy()

pos_scaler = StandardScaler()
X_train_pos = pos_scaler.fit_transform(X_train_pos)
X_test_pos = pos_scaler.transform(X_test_pos)

X_train_pos_tensor = torch.tensor(X_train_pos, dtype=torch.float32).to(device)
X_test_pos_tensor = torch.tensor(X_test_pos, dtype=torch.float32).to(device)

class PlayerWorthPredictor(nn.Module):
    def __init__(self):
        super(PlayerWorthPredictor, self).__init__()
        self.fc1 = nn.Linear(X_train_worth_tensor.shape[1], 64)
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

model_worth = PlayerWorthPredictor().to(device)
model_pos = PositionPredictor(18, 15).to(device)

model_worth.load_state_dict(torch.load('./model/worth_model_file.pth'))
model_pos.load_state_dict(torch.load('./model/pos_model_file.pth'))

# Predict player worth for the training data
model_worth.eval()
with torch.no_grad():
    predicted_worth_train = model_worth(X_train_worth_tensor).cpu().numpy()

# Predict player position for the training data
model_pos.eval()
with torch.no_grad():
    predicted_positions_train = model_pos(X_train_pos_tensor).cpu().numpy().argmax(axis=1)

# Create a DataFrame with predictions and actual values for training data
worth_data_train['Predicted_Worth'] = predicted_worth_train
worth_data_train['Predicted_Position'] = predicted_positions_train

# Group by position and calculate mean worth for the training data
actual_worth_by_position_train = worth_data_train.groupby('Predicted_Position')['Value'].mean()
predicted_worth_by_position_train = worth_data_train.groupby('Predicted_Position')['Predicted_Worth'].mean()

# Define position labels
position_labels = ['ST', 'RW', 'LW', 'CDM', 'CB', 'RM', 'CM', 'LM', 'LB', 'CAM', 'RB', 'CF', 'RWB', 'LWB', 'GK']
positions = actual_worth_by_position_train.index
actual_worth_values_train = [actual_worth_by_position_train[pos] for pos in positions]

# Define a range of scaling factors to test
scaling_factors = np.linspace(0.5, 1.5, 100)

# Initialize a dictionary to store the best scaling factor for each position
best_scaling_factors = {}

# Initialize a dictionary to store the minimum error for each position
min_errors = {position: float('inf') for position in range(len(position_labels))}

# Iterate over all scaling factors
for scaling_factor in scaling_factors:
    # Apply the scaling factor to the predicted worth
    worth_data_train['Scaled_Predicted_Worth'] = worth_data_train['Predicted_Worth'] * scaling_factor

    # Group by predicted position and calculate the mean worth
    scaled_predicted_worth_by_position_train = worth_data_train.groupby('Predicted_Position')['Scaled_Predicted_Worth'].mean()

    # Calculate the mean squared error for each position
    for position in scaled_predicted_worth_by_position_train.index:
        actual_worth = actual_worth_by_position_train.loc[position]
        scaled_predicted_worth = scaled_predicted_worth_by_position_train.loc[position]
        error = (actual_worth - scaled_predicted_worth) ** 2

        # Update the best scaling factor if the error is lower than the current minimum error
        if error < min_errors[position]:
            min_errors[position] = error
            best_scaling_factors[position] = scaling_factor

# Print the best scaling factors for each position
for position, scaling_factor in best_scaling_factors.items():
    print(f"Position: {position_labels[position]}, Best Scaling Factor: {scaling_factor}")


# Predict player worth for the test data
with torch.no_grad():
    predicted_worth_test = model_worth(X_test_worth_tensor).cpu().numpy()

# Predict player position for the test data
with torch.no_grad():
    predicted_positions_test = model_pos(X_test_pos_tensor).cpu().numpy().argmax(axis=1)

# Create a DataFrame with predictions and actual values for test data
worth_data_test['Predicted_Worth'] = predicted_worth_test
worth_data_test['Predicted_Position'] = predicted_positions_test

# Apply the best scaling factors to the predicted worth for the test data
worth_data_test['Scaled_Predicted_Worth'] = worth_data_test.apply(
    lambda row: row['Predicted_Worth'] * best_scaling_factors.get(row['Predicted_Position'], 1.0),
    axis=1
)

# Calculate the test MSE
test_mse = mean_squared_error(worth_data_test['Value'], worth_data_test['Scaled_Predicted_Worth'])
print(f"Total Test MSE: {test_mse}")

# Group by position and calculate mean worth for the test data
actual_worth_by_position_test = worth_data_test.groupby('Predicted_Position')['Value'].mean()
scaled_predicted_worth_by_position_test = worth_data_test.groupby('Predicted_Position')['Scaled_Predicted_Worth'].mean()

# Ensure positions and worth values are aligned
positions = np.arange(len(position_labels))
actual_worth_values_test = [actual_worth_by_position_test.get(pos, 0) for pos in positions]
scaled_worth_values_test = [scaled_predicted_worth_by_position_test.get(pos, 0) for pos in positions]

# Plot the scaled worths using the best scaling factors for the test data
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.35  # width of the bars

# Ensure lengths of position_labels and worth values match
assert len(position_labels) == len(actual_worth_values_test) == len(scaled_worth_values_test), "Mismatch in lengths"

ax.bar(np.arange(len(position_labels)) - width / 2, actual_worth_values_test, width, label='Actual Worth')
ax.bar(np.arange(len(position_labels)) + width / 2, scaled_worth_values_test, width, label='Scaled Predicted Worth')

ax.set_xlabel('Player Position')
ax.set_ylabel('Worth')
ax.set_title('Actual vs Scaled Predicted Player Worth by Position (Test Data)')
ax.legend()

plt.xticks(np.arange(len(position_labels)), [position_labels[i] for i in range(len(position_labels))])
plt.show()
