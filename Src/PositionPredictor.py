import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader # type: ignore
from sklearn.metrics import precision_recall_fscore_support
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
from tqdm import tqdm
from torchsummary import summary
from sympy import false
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

# hyperparameters
learning_rate = 0.0001
batch_size = 128
num_epochs = 500
val_size = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('badadataset.csv', encoding='unicode escape', low_memory=False)

input_features = ['Acceleration', 'Aggression', 'Agility', 'Ball control', 'Curve', 'Dribbling',
                  'Finishing', 'FK Accuracy', 'Heading accuracy', 'Interceptions',
                  'Jumping', 'Long shots', 'Penalties', 'Physical / Positioning',
                  'Shot power', 'Strength', 'Vision', 'Volleys',
                  'Sliding tackle', 'Best position']
data = data[input_features]

l2 = data.columns
l = []
lis = []
for i in range(0, len(data.columns)):
    if (l2[i] == 'Best position'): continue
    for j in range(0, len(data['Best position'].values)):
        try:
            int_a = eval(str(data.iloc[j, i]))
            data.iloc[j, i] = int_a
        except NameError:
            if (j not in l):
                l.append(j)
        except TypeError:
            if (j not in l):
                l.append(j)

data = data.drop(l)
# pd.to_csv(data, 'CompleteDatasetmayankcorrect.csv')
data.dropna()
data = data.reset_index()

df = pd.DataFrame(data['Best position'])
df['Best position'] = df['Best position'].str.split()  # Ensure it is a list of positions

# Define the columns representing each position
cols = ['ST', 'RW', 'LW', 'CDM', 'CB', 'RM', 'CM', 'LM', 'LB', 'CAM', 'RB', 'CF', 'RWB', 'LWB', 'GK']

# Initialize the new dataframe with zeros
newdf = pd.DataFrame(0, columns=cols, index=range(len(df)))

# Populate the new dataframe based on the Best position
for idx, row in df.iterrows():
    for col in cols:
        if col in row['Best position']:
            newdf.at[idx, col] = 1

newdf = newdf.reset_index(drop=True)

input = ['Aggression', 'Agility', 'Ball control', 'Curve', 'Dribbling', 'Finishing'
    , 'FK Accuracy', 'Heading accuracy', 'Interceptions',
         'Jumping', 'Long shots', 'Penalties', 'Physical / Positioning',
         'Shot power', 'Strength', 'Vision', 'Volleys',
         'Sliding tackle']

X = data[input]  # Include only the columns you want as input features
y = newdf

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)


class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_dataset = Dataset(X_train_tensor, y_train_tensor)
test_dataset = Dataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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


prediction = Classifier(18, 15).to(device)
optimizer = optim.AdamW(params=prediction.parameters(), lr=0.001, weight_decay=0.001)
loss_function = nn.NLLLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prediction = prediction.to(device)
# Initialize record dictionary to store metrics
record = {
    "Train Loss": [],
    "Train Accuracy": [],
    "Test Loss": [],
    "Test Accuracy": []
}

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

for epoch in tqdm(range(num_epochs), leave=False):
    prediction.train()
    train_loss = 0
    train_correct = 0

    for idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = prediction(x_batch)
        loss = loss_function(output, y_batch.argmax(dim=1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (output.argmax(1) == y_batch.argmax(1)).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = train_correct / len(train_loader.dataset) * 100 # type: ignore

    with torch.no_grad():
        prediction.eval()
        val_loss = 0
        val_correct = 0

        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = prediction(x_batch)
            loss = loss_function(output, y_batch.argmax(dim=1))
            val_loss += loss.item()
            val_correct += (output.argmax(1) == y_batch.argmax(1)).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset) * 100 # type: ignore

    record["Train Loss"].append(train_loss)
    record["Train Accuracy"].append(train_accuracy)
    record["Test Loss"].append(val_loss)
    record["Test Accuracy"].append(val_accuracy)

    scheduler.step(val_loss)  # LR scheduler

    if epoch % 1000 == 0:
        tqdm.set_description(f"Epoch [{epoch}/{num_epochs}]") # type: ignore
        tqdm.set_postfix(train_loss=train_loss, train_acc=train_accuracy, test_loss=val_loss, test_acc=val_accuracy) # type: ignore

# joblib.dump(prediction, '../model/pospredfin.pkl')
# joblib.dump(scaler, '../model/poscalerfin.pkl')
torch.save(prediction.state_dict(), './model/pos_model_file.pth')
