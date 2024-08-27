import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
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
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

# Hyperparameters
learning_rate = 0.0001
batch_size = 128
num_epochs = 100
val_size = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data = pd.read_csv('./Data/MergedData.csv', encoding='unicode escape', low_memory=False)
data = data.dropna()
data.reset_index(drop=True, inplace=True)

# Select input features and target column
input_features = ['Acceleration', 'Aggression', 'Agility', 'BallControl', 'Curve', 'Dribbling',
                  'Finishing', 'FKAccuracy', 'HeadingAccuracy', 'Interceptions',
                  'Jumping', 'LongShots', 'Penalties', 'Positioning',
                  'ShotPower', 'Strength', 'Vision', 'Volleys',
                  'SlidingTackle', 'StandingTackle', 'Best Position']

data = data[input_features]

# Process target column (Best Position)
df = pd.DataFrame(data['Best Position'])
# df['Best Position'] = df['Best Position'].str.split()

# Encode labels

enc = ['ST', 'RW', 'LW', 'CDM', 'CB', 'RM', 'CM', 'LM', 'LB', 'CAM', 'RB', 'CF', 'RWB', 'LWB', 'GK']
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

def encode(x):
    return enc.index(x)

y_encoded = df['Best Position'].apply(encode)

# Select input features for the model
input = ['Aggression', 'Agility', 'BallControl', 'Curve', 'Dribbling', 'Finishing',
         'FKAccuracy', 'HeadingAccuracy', 'Interceptions',
         'Jumping', 'LongShots', 'Penalties', 'Positioning',
         'ShotPower', 'Strength', 'Vision', 'Volleys',
         'SlidingTackle', 'StandingTackle']

X = data[input]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
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

# Initialize model, optimizer, and loss function
num_classes = len(enc)
prediction = Classifier(19, num_classes).to(device)
optimizer = optim.AdamW(params=prediction.parameters(), lr=learning_rate, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# Initialize record dictionary to store metrics
record = {
    "Train Loss": [],
    "Train Accuracy": [],
    "Test Loss": [],
    "Test Accuracy": []
}

# Training loop
for epoch in range(num_epochs):
    prediction.train()
    train_loss = 0
    train_correct = 0
    loop = tqdm(train_loader)

    for idx, (x_batch, y_batch) in enumerate(loop):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = prediction(x_batch)
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (output.argmax(1) == y_batch).sum().item()

        if idx % 10 == 0:
            loop.set_description(f"Epoch: [{epoch + 1}/{num_epochs}]")

    train_loss /= len(train_loader)
    train_accuracy = train_correct / len(train_loader.dataset) * 100 # type: ignore

    with torch.no_grad():
        prediction.eval()
        val_loss = 0
        val_correct = 0

        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = prediction(x_batch)
            loss = loss_function(output, y_batch)
            val_loss += loss.item()
            val_correct += (output.argmax(1) == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset) * 100 # type: ignore

    record["Train Loss"].append(train_loss)
    record["Train Accuracy"].append(train_accuracy)
    record["Test Loss"].append(val_loss)
    record["Test Accuracy"].append(val_accuracy)

    scheduler.step(val_loss)

    print(f"Epoch: [{epoch + 1}/{num_epochs}] || train_loss: {train_loss:.4f} || val_loss: {val_loss:.4f}", end="")
    print(f" || train_acc: {train_accuracy:.2f} || val_acc: {val_accuracy:.2f}")


# Plot and save the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(record["Train Loss"], label="Train Loss")
plt.plot(record["Test Loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig('./plots/loss_plot.png')
plt.show()

plt.clf()

# Plot and save the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(record["Train Accuracy"], label="Train Accuracy")
plt.plot(record["Test Accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig('./plots/accuracy_plot.png')
plt.show()


# Save the model, scaler, and label encoder
torch.save(prediction.state_dict(), './model/pos_model_file.pth')
joblib.dump(scaler, './model/poscalerfin.pkl')
# joblib.dump(label_encoder, './model/label_encoder.pkl')
