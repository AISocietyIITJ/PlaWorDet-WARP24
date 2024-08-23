import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import json
import joblib


class PlayerWorthPredictor(nn.Module):
    def __init__(self):
        super(PlayerWorthPredictor, self).__init__()
        self.fc1 = nn.Linear(37, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class predictor(nn.Module):

    def __init__(self):
        super(predictor, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.worthpred = PlayerWorthPredictor()
        self.pospred = Classifier(18, 15)

        # Loading only the model weights
        self.worthpred.load_state_dict(torch.load('./model/worth_model_file.pth', weights_only=True))
        self.pospred.load_state_dict(torch.load('./model/pos_model_file.pth', weights_only=True))

        self.name_data = pd.read_csv('./Data/CompleteDatasetmayank.csv', encoding='ISO-8859-1', low_memory=False)

        data = pd.read_csv('./Data/worth_data.csv')

        worth_data = data.drop('index', axis=1)

        X_worth = worth_data.drop('Value', axis=1)
        y_worth = worth_data['Value']

        # worth_scaler = StandardScaler()
        worth_scaler = joblib.load('./model/worth_scaler.pkl')
        X_worth = worth_scaler.transform(X_worth)

        self.X_worth_tensor = torch.tensor(X_worth, dtype=torch.float32).to(device)
        self.y_worth_tensor = torch.tensor(y_worth.values, dtype=torch.float32).view(-1, 1).to(device)

        # Creating position_data
        pos_data = data

        input_features = ['Aggression', 'Agility', 'Ball control', 'Curve', 'Dribbling', 'Finishing',
                          'FK accuracy', 'Heading accuracy', 'Interceptions',
                          'Jumping', 'Long shots', 'Penalties', 'Physical / Positioning',
                          'Shot power', 'Strength', 'Vision', 'Volleys',
                          'Slide Tack']

        X_pos = pos_data[input_features]

        # pos_scaler = StandardScaler()
        pos_scaler = joblib.load('./model/pos_scaler.pkl')

        X_pos = pos_scaler.transform(X_pos)
        self.X_pos_tensor = torch.tensor(X_pos, dtype=torch.float32).to(device)

        # Load JSON data from the file
        with open('./model/best_scaling_factors.json', 'r') as file:
            self.scaling_factors = json.load(file)

    def forward(self, name):

        idx = self.find_index(name)
        self.worthpred.eval()
        self.pospred.eval()
        pos_data = self.X_pos_tensor[idx].unsqueeze(0)
        worth_data = self.X_worth_tensor[idx].unsqueeze(0)

        pos_probs = self.pospred(pos_data)
        pos = torch.argmax(pos_probs, dim=1)
        worth = self.worthpred(worth_data)

        return self.y_worth_tensor[idx].item(), (worth * self.scaling_factors[str(pos.item())]).item()

    def find_index(self, name):
        matching_rows = self.name_data.index[self.name_data['Name'] == name].tolist()

        if not matching_rows:
            raise ValueError(f"Player '{name}' not found in the dataset.")

        # Return the first matching index (in case there are duplicates)
        return matching_rows[0]

if __name__ == '__main__':
    pred = predictor()
    print(pred('Neymar'))
