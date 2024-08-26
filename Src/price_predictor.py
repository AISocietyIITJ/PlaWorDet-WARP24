import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from Similarity_Score import sim_func
import json
import joblib


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 8
        self.out_features = 1

        self.my_network = torch.nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout1d(0.1),

            nn.Linear(128, 1024),
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


class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.my_network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout1d(0.1),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout1d(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, out_features),
        )

    def forward(self, x):
        return self.my_network(x)


class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.worthpred = Model()
        self.pospred = Classifier(19, 15)

        # Loading only the model weights
        self.worthpred.load_state_dict(torch.load('./model/app2_f1.pth', weights_only=True))
        self.pospred.load_state_dict(torch.load('./model/pos_model_file.pth', weights_only=True))

        data = pd.read_csv('./Data/MergedData.csv')
        data.reset_index(drop=True, inplace=True)

        position_input_features = ['Aggression', 'Agility', 'BallControl', 'Curve', 'Dribbling', 'Finishing',
                                   'FKAccuracy', 'HeadingAccuracy', 'Interceptions',
                                   'Jumping', 'LongShots', 'Penalties', 'Positioning',
                                   'ShotPower', 'Strength', 'Vision', 'Volleys',
                                   'SlidingTackle', 'StandingTackle']

        pos_data = data[position_input_features].copy()

        pos_scaler = StandardScaler()
        pos_data = pos_scaler.fit_transform(pos_data)

        pos_data_tensor = torch.tensor(pos_data, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            self.pospred.eval()
            pos_data_tensor = pos_data_tensor.to(DEVICE)
            predicted_positions = self.pospred(pos_data_tensor).cpu().numpy().argmax(axis=1)

        final_input_data = data[['Dribbling', 'LongPassing', 'ShortPassing',
                                 'Overall', 'Special', 'BallControl', 'ShotPower', 'Finishing', 'Value']]

        self.name_data = data[['ID', 'Name']]

        # final_input_data = data[['Sprint speed', 'Dribbling', 'Shot power', 'Reactions',
        #     'Long passing', 'Short passing', 'Physical / Positioning', 'Value']]

        final_input_data['Passing'] = (final_input_data['LongPassing'] + final_input_data['ShortPassing']) / 2
        final_input_data = final_input_data.drop(['LongPassing', 'ShortPassing'], axis=1)
        final_input_data['Position'] = predicted_positions

        self.y_data_tensor = torch.tensor(final_input_data['Value'], dtype=torch.float32).view(-1, 1).to(DEVICE)
        final_input_data = final_input_data.drop(['Value'], axis=1)

        scaler = StandardScaler()
        X_data_tensor = scaler.fit_transform(final_input_data)
        self.X_data_tensor = torch.tensor(X_data_tensor, dtype=torch.float32).to(DEVICE)


    def find_similarity_score(self, name):
        id, club = self.find_id_club(name)

        try:
            similarity_score = sim_func(id, club)
            return similarity_score
        except ValueError:
            print('Data not found')

    def find_base_price(self, idx):

        self.worthpred.eval()
        self.pospred.eval()
        worth_data = self.X_data_tensor[idx].unsqueeze(0)
        worth = self.worthpred(worth_data)

        return np.exp(worth.item()), self.y_data_tensor[idx].item()

    def find_index(self, name):
        matching_rows = self.name_data.index[self.name_data['Name'] == name].tolist()

        if not matching_rows:
            raise ValueError(f"Player '{name}' not found in the dataset.")

        # Return the first matching index (in case there are duplicates)
        return matching_rows[0]

    def find_id_club(self, name):
        index = self.find_index(name)
        return self.name_data[index]['ID'], self.name_data[index]['Club']


if __name__ == '__main__':
    pred = predictor()
    print(pred(9))
