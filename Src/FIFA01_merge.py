import pandas as pd

# Load all the datasets
df17 = pd.read_csv('./Data/FIFA17_official_data.csv', index_col=None, header=0)
df18 = pd.read_csv('./Data/FIFA18_official_data.csv', index_col=None, header=0)
df19 = pd.read_csv('./Data/FIFA19_official_data.csv', index_col=None, header=0)
df20 = pd.read_csv('./Data/FIFA20_official_data.csv', index_col=None, header=0)
df21 = pd.read_csv('./Data/FIFA21_official_data.csv', index_col=None, header=0)
df22 = pd.read_csv('./Data/FIFA22_official_data.csv', index_col=None, header=0)
df23 = pd.read_csv('./Data/FIFA23_official_data.csv', index_col=None, header=0)

# Concatenate all the datasets
df = pd.concat([df17, df18, df19, df20, df21, df22, df23], axis=0)
print(df.shape)
# Define the features to keep
features = ['ID', 'Name', 'Club', 'Age', 'Overall', 'Special', 'Acceleration', 'Aggression', 'Agility', 'BallControl', 'Curve', 'Dribbling', 'Finishing',
                  'FKAccuracy', 'HeadingAccuracy', 'Interceptions', 'Positioning',
                  'Jumping', 'LongPassing', 'ShortPassing', 'LongShots', 'Penalties',
                  'ShotPower', 'Strength', 'Vision', 'Volleys', 'Best Position',
                  'SlidingTackle', 'StandingTackle', 'Finishing', 'Value']

# Create a copy with the selected features
df = df[features].copy()
print(df.shape)

# Drop rows with missing values
df = df.dropna()

# Convert 'Value' from string to numeric
def convert_to_number(x):
    if 'M' in x:
        return float(x.replace('M', ''))
    elif 'K' in x:
        return float(x.replace('K', '')) / 1_000
    else:
        return float(x)

df['Value'] = df['Value'].str.replace('€', '', regex=False).astype(str)
df['Value'] = df['Value'].apply(convert_to_number)

# Filter out rows where 'Value' is 0
df = df[df['Value'] != 0]

# Reset the index and add it as a column
df.reset_index(drop=True, inplace=True)
df['Index'] = df.index

print(df.columns)
print(df.shape)

# Save the final DataFrame to a CSV file
df.to_csv('./Data/MergedData.csv', index=False)
