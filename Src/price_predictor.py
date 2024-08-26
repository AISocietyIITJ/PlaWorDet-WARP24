import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from Similarity_Score import sim_func
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 8
        self.out_features = 1

        self.my_network = torch.nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

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

        self.worthpred = Model().to(DEVICE)
        self.pospred = Classifier(19, 15).to(DEVICE)

        # Loading only the model weights
        self.worthpred.load_state_dict(torch.load('./model/app2_f1.pth', map_location=DEVICE))
        self.pospred.load_state_dict(torch.load('./model/pos_model_file.pth', map_location=DEVICE))

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

        self.name_data = data[['Index', 'ID', 'Name', 'Club']]

        final_input_data['Passing'] = (final_input_data['LongPassing'] + final_input_data['ShortPassing']) / 2
        final_input_data = final_input_data.drop(['LongPassing', 'ShortPassing'], axis=1)
        final_input_data['Position'] = predicted_positions

        self.total_data = final_input_data

        self.y_data_tensor = torch.tensor(final_input_data['Value'].values, dtype=torch.float32).view(-1, 1).to(DEVICE)
        final_input_data = final_input_data.drop(['Value'], axis=1)

        self.pos_encoder = LabelEncoder()
        self.pos_encoder = joblib.load('./model/label_encoder.pkl')

        player_data = pd.read_csv('./Data/ProcessedData.csv')
        player_data = player_data.dropna()
        player_data.reset_index(drop=True, inplace=True)

        # Prepare the features and target variable
        X = player_data[['Predicted Price', 'Similarity Score']]
        y = player_data['Actual Price']

        base_learners = [
            ('linear', LinearRegression()),
            ('ridge', Ridge()),
            ('lasso', Lasso(max_iter=10000)),
            ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=100))
        ]

        self.stacking_model = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
        self.stacking_model.fit(X, y)

        self.scaler = StandardScaler()
        X_data = self.scaler.fit_transform(final_input_data)
        self.X_data_tensor = torch.tensor(X_data, dtype=torch.float32).to(DEVICE)

    def find_similarity_score(self, name, club):
        id = self.find_id(name)

        try:
            similarity_score = sim_func(id, club)
            return similarity_score
        except ValueError:
            print('Data not found')
            return None

    def find_base_price(self, idx, position):
        self.worthpred.eval()
        # Select the data for the specific player index
        data = self.total_data.iloc[idx].copy()
        # Encode the position and update the data
        encoded_position = self.pos_encoder.transform([position])[0]
        data['Position'] = encoded_position
        # Drop the 'Value' column to get the input features
        X_data = data.drop(['Value'], axis=0)
        # Convert to DataFrame to perform scaling
        X_data = pd.DataFrame([X_data])
        # Scale the data
        X_data_scaled = self.scaler.transform(X_data)
        # Convert to tensor
        X_data_tensor = torch.tensor(X_data_scaled, dtype=torch.float32).to(DEVICE)
        # Predict the worth using the neural network
        with torch.no_grad():
            worth = self.worthpred(X_data_tensor).item()

        # Get the actual value (unscaled)
        actual_price = np.exp(data['Value'])
        return np.exp(worth), self.y_data_tensor[idx].item()

    def find_index(self, name):
        matching_rows = self.name_data.index[self.name_data['Name'] == name].tolist()

        if not matching_rows:
            raise ValueError(f"Player '{name}' not found in the dataset.")

        print(matching_rows)
        # Return the first matching index (in case there are duplicates)
        return matching_rows[0]

    def find_id(self, name):
        index = self.find_index(name)
        return self.name_data.iloc[index]['ID']

    def calculate_final_price(self, predicted_price, similarity_score):
        # Convert the inputs into a DataFrame with the same structure as the training data
        input_data = pd.DataFrame({
            'Predicted Price': [predicted_price],
            'Similarity Score': [similarity_score]
        })

        # Use the stacking model to predict the final price
        final_price = self.stacking_model.predict(input_data)

        return final_price[0]

    def forward(self, name, club, position):
        predicted_price, actual_price = self.find_base_price(self.find_index(name), position)
        similarity_score = self.find_similarity_score(name, club)
        if similarity_score is None:
            return None, actual_price
        final_price = self.calculate_final_price(predicted_price, similarity_score)
        return final_price, actual_price, self.find_index(name)

if __name__ == '__main__':
    pred = predictor()
    print(pred('L. Messi', 'FC Barcelona', 'ST'))
