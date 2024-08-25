import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from skimage.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4

data = pd.read_csv('./Data/worth_data.csv')
data.reset_index(drop=True, inplace=True)

position_input_features = ['Aggression', 'Agility', 'Ball control', 'Curve', 'Dribbling', 'Finishing',
                  'FK accuracy', 'Heading accuracy', 'Interceptions',
                  'Jumping', 'Long shots', 'Penalties', 'Physical / Positioning',
                  'Shot power', 'Strength', 'Vision', 'Volleys',
                  'Slide Tack']

pos_data = data[position_input_features].copy()

pos_scaler = StandardScaler()
pos_data = pos_scaler.fit_transform(pos_data)

pos_data_tensor = torch.tensor(pos_data, dtype=torch.float32).to(DEVICE)


class Classifier(nn.Module):

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

pos_classifier = Classifier(18, 15).to(DEVICE)
pos_classifier.load_state_dict(torch.load('./model/pos_model_file.pth'))

with torch.no_grad():
    pos_classifier.eval()
    pos_data_tensor = pos_data_tensor.to(DEVICE)
    predicted_positions = pos_classifier(pos_data_tensor).cpu().numpy().argmax(axis=1)

# final_input_data = data[['Dribbling', 'Long passing', 'Short passing',
#                          'Overall', 'Special', 'Ball control', 'Shot power', 'Finishing', 'Value']]

final_input_data = data[['Aggression', 'Interceptions', 'Sprint speed', 'Dribbling', 'Shot power', 'Reactions',
    'Long passing', 'Short passing', 'Physical / Positioning', 'Value']]

final_input_data['Passing'] = (final_input_data['Long passing'] + final_input_data['Short passing']) / 2
final_input_data = final_input_data.drop(['Long passing', 'Short passing'], axis=1)
final_input_data['Position'] = predicted_positions

y_data_tensor = torch.tensor(final_input_data['Value'], dtype=torch.float32).view(-1, 1).to(DEVICE)

final_input_data['Value'] = np.log(final_input_data['Value'])

# print(final_input_data.head(10))
# exit(0)

final_output_data = final_input_data['Value']
final_input_data = final_input_data.drop(['Value'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(final_input_data, final_output_data
                                                    , test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_data_tensor = scaler.transform(final_input_data)
X_data_tensor = torch.tensor(X_data_tensor, dtype=torch.float32).to(DEVICE)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(DEVICE)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(DEVICE)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset =test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print(X_train_tensor.shape)
# print(y_train_tensor.shape)
# exit()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = X_train.shape[1]
        self.out_features = 1

        self.my_network = torch.nn.Sequential(
            nn.Linear(self.in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout1d(0.1),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Dropout1d(0.2),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout1d(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout1d(0.2),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.my_network(x)

model = Model().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
criterion = nn.MSELoss()

record = {
    "Train Loss": [],
    "Val Loss": [],
}

for epoch in range(NUM_EPOCHS):
    model.train()
    train_predictions = []
    train_targets = []
    train_loss = 0.0
    for data, target in train_loader:
        data = data.reshape(BATCH_SIZE, -1).to(DEVICE)
        target = target.reshape(BATCH_SIZE, -1).to(DEVICE)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()*data.size(0)
        train_predictions.extend(output.cpu().detach().numpy())
        train_targets.extend(target.cpu().detach().numpy())

    # train_loss /= len(train_loader.dataset)
    train_loss = mean_squared_error(train_targets, train_predictions)

    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # val_loss += loss.item() * inputs.size(0)
            val_predictions.extend(outputs.cpu().detach().numpy())
            val_targets.extend(targets.cpu().detach().numpy())

            scheduler.step(train_loss)

    # val_loss /= len(val_loader.dataset)
    val_loss = mean_squared_error(val_targets, val_predictions)

    record["Train Loss"].append(train_loss)
    record["Val Loss"].append(val_loss)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    # Calculate R2 score for training set
    train_r2 = r2_score(train_predictions, train_targets)

    # Calculate R2 score for validation set
    val_r2 = r2_score(val_predictions, val_targets)

    print(f"Training R2 Score: {train_r2:.4f}")
    print(f"Validation R2 Score: {val_r2:.4f}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


def predict_player_value(index):

    # Extract player data at the given index
    player_input = X_data_tensor[index].unsqueeze(0)

    # Predict the player's value
    model.eval()
    with torch.no_grad():
        predicted_value = model(player_input).cpu().numpy()

    # Return the predicted value (exponential of log value)
    return np.exp(predicted_value[0][0]), y_data_tensor[index]

print(predict_player_value(144))
print(predict_player_value(4101))
print(predict_player_value(6984))
print(predict_player_value(6708))
print(predict_player_value(140))
# print(final_output_data.head())

# torch.save(model.state_dict(), './model/app2_f2.pth')

plt.plot(record["Train Loss"], label="Train")
plt.plot(record["Val Loss"], label="Val")
plt.legend()

# plt.savefig('plot_2.png', dpi=300, bbox_inches='tight')

plt.show()
