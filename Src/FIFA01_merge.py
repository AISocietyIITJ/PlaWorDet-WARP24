import pandas as pd

df17 = pd.read_csv('./Data/FIFA17_official_data.csv', index_col=0, header=0)
df18 = pd.read_csv('./Data/FIFA18_official_data.csv', index_col=0, header=0)
df19 = pd.read_csv('./Data/FIFA19_official_data.csv', index_col=0, header=0)
df20 = pd.read_csv('./Data/FIFA20_official_data.csv', index_col=0, header=0)
df21 = pd.read_csv('./Data/FIFA21_official_data.csv', index_col=0, header=0)
df22 = pd.read_csv('./Data/FIFA22_official_data.csv', index_col=0, header=0)
df23 = pd.read_csv('./Data/FIFA23_official_data.csv', index_col=0, header=0)

df = pd.concat([df17, df18, df19, df20, df21, df22, df23], axis=0)
print(df.shape)

features = ['Name', 'Club', 'Age', 'Overall', 'Special', 'Acceleration', 'Aggression', 'Agility', 'BallControl', 'Curve', 'Dribbling', 'Finishing',
                  'FKAccuracy', 'HeadingAccuracy', 'Interceptions', 'Positioning',
                  'Jumping', 'LongPassing', 'ShortPassing', 'LongShots', 'Penalties',
                  'ShotPower', 'Strength', 'Vision', 'Volleys', 'Best Position',
                  'SlidingTackle', 'StandingTackle', 'Finishing', 'Value']

df = df[features].copy()
print(df.shape)
df = df.dropna()
def convert_to_number(x):
    if 'M' in x:
        return float(x.replace('M', ''))
    elif 'K' in x:
        return float(x.replace('K', '')) / 1_000
    else:
        return float(x)


df['Value'] = df['Value'].str.replace('€', '', regex=False).astype(str)

# Apply the conversion
df['Value'] = df['Value'].apply(convert_to_number)

df = df[df['Value'] != 0]

print(df.shape)

df.to_csv('./Data/MergedData.csv')
