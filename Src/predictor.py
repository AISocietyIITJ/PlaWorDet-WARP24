import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import json
import joblib
import worthpredictionbasic
import PositionPredictor


class predictor(nn.Module):

    def __init__(self):
        super(predictor, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.worthpred = worthpredictionbasic.PlayerWorthPredictor()
        self.pospred = PositionPredictor.Classifier(18, 15)

        self.worthpred.load_state_dict(torch.load('./model/worth_model_file.pth'))
        self.pospred.load_state_dict(torch.load('./model/pos_model_file.pth'))

        data = pd.read_csv('./Data/worth_data.csv')

        worth_data = data.drop('index', axis=1)

        X_worth = worth_data.drop('Value', axis=1)
        y_worth = worth_data['Value']

        # worth_scaler = StandardScaler()
        worth_scaler = joblib.load('./model/worth_scaler.pkl')
        X_worth = worth_scaler.transform(X_worth)

        X_worth_tensor = torch.tensor(X_worth, dtype=torch.float32).to(device)
        y_worth_tensor = torch.tensor(y_worth.values, dtype=torch.float32).view(-1, 1).to(device)

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
        X_pos_tensor = torch.tensor(X_pos, dtype=torch.float32).to(device)

        # Load JSON data from the file
        scaling_factors = {}
        with open('./model/best_scaling_factors.json', 'r') as file:
            scaling_factors = json.load(file)

    def forward(self, x):
        worth = self.worthpred(x)
        pos = self.pospred(x)
        return worth * self.scaling_factors[pos]
