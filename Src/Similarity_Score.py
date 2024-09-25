import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
from numpy import inf
from sklearn.preprocessing import normalize
import pickle
import re
import warnings


warnings.filterwarnings('ignore')

df_17 = pd.read_csv('./Data/FIFA17_official_data.csv')
df_18 = pd.read_csv('./Data/FIFA18_official_data.csv')
df_19 = pd.read_csv('./Data/FIFA19_official_data.csv')
df_20 = pd.read_csv('./Data/FIFA20_official_data.csv')
df_21 = pd.read_csv('./Data/FIFA21_official_data.csv')
df_22 = pd.read_csv('./Data/FIFA22_official_data.csv')

import pandas as pd
import numpy as np


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces NaN or null values in numeric columns of a DataFrame with the mean or mode.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with NaN or null values replaced.
    """
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Calculate mean and mode
            mean_value = df[column].mean()
            mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else np.nan

            # Replace NaN with mean if available, otherwise use mode
            if not np.isnan(mean_value):
                df[column].fillna(mean_value, inplace=True)
            else:
                df[column].fillna(mode_value, inplace=True)

    return df


df_17 = fill_missing_values(df_17)
df_18 = fill_missing_values(df_18)
df_19 = fill_missing_values(df_19)
df_20 = fill_missing_values(df_20)
df_21 = fill_missing_values(df_21)
df_22 = fill_missing_values(df_22)

datalist = [df_17,df_18,df_19,df_20,df_21,df_22]

curr_set = df_17
for next_set in datalist[1:]:
    curr_set = next_set

for dataset in datalist:
    if 'Release Clause' in list(dataset.columns):
        dataset.drop("Release Clause", axis = 1, inplace = True)
    if 'DefensiveAwareness' in list(dataset.columns):
        dataset.drop("DefensiveAwareness", axis = 1, inplace = True)

curr_set = df_17

F_all_df = pd.concat(datalist, ignore_index=True)

Unimportant_features = ["Photo", "Flag", "Club Logo", "Wage", "Special", "International Reputation", "Work Rate", "Body Type", "Real Face", "Jersey Number", "Contract Valid Until", "Best Overall Rating","Joined","Loaned From"]
F_all_df.drop(Unimportant_features, axis = 1, inplace=True)
F_all_df["Ovr_pot"] = (F_all_df["Overall"] + F_all_df["Potential"]) / 2

F_all_df = pd.get_dummies(F_all_df, columns = ["Preferred Foot"])

detailed_features = ['Age','Overall', 'Potential', 'Weak Foot', 'Skill Moves', 'Height', 'Weight', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes', "Ovr_pot", 'Preferred Foot_Left', 'Preferred Foot_Right']

F_all_df[detailed_features] = F_all_df[detailed_features].fillna(0)

feet_inches_re = re.compile(r"(\d)'(\d+)")

def feet_inches_to_cm(s):
    match = feet_inches_re.match(s)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        return round((feet*12 + inches) * 2.54, 2)
    else:
        return float(s.replace('cm', ''))
F_all_df['Height'] = pd.to_numeric(F_all_df['Height'].apply(feet_inches_to_cm))

def convert_weight(weight_str):
    if 'kg' in weight_str:
        return float(weight_str.replace('kg', ''))
    elif 'lbs' in weight_str:
        return float(weight_str.replace('lbs', '')) * 0.453592
    else:
        print("oops")
        return pd.np.nan


F_all_df['Weight'] = F_all_df['Weight'].apply(convert_weight)

def convert_cost(cost_str):
    if 'M' in cost_str:
        return float(cost_str.replace('€', '').replace('M', ''))
    elif 'K' in cost_str:
        return float(cost_str.replace('€', '').replace('K', '')) / 1000


F_all_df['Value'] = F_all_df['Value'].apply(convert_cost)

scaler = sklearn.preprocessing.StandardScaler()
F_norm_df = F_all_df.copy(deep = True)
F_norm_df[detailed_features] = scaler.fit_transform(F_norm_df[detailed_features].to_numpy())

df_22 = F_all_df.tail(df_22.shape[0])
df_21 = F_all_df.tail(df_21.shape[0])
df_20 = F_all_df.tail(df_20.shape[0])
df_19 = F_all_df.tail(df_19.shape[0])
df_18 = F_all_df.tail(df_18.shape[0])
df_17 = F_all_df.tail(df_17.shape[0])

def get_pos_weights(pos_df,pos_essential_ft):
    pos_weights_per_club={}

    for club in all_clubs:

        all_features = pos_df[pos_df["Club"] == club]
        all_features = all_features.sort_values('Ovr_pot',ascending = False).head(20)
        essential_features = all_features[pos_essential_ft]
        weights_vector=[]
        sum = 0


        for i in essential_features:
            if essential_features[i].std() != 0:
                weights_vector.append(1/essential_features[i].std())
                sum += 1/essential_features[i].std()
            else:
                weights_vector.append(0)
                sum += 0
        if(sum != 0):
            for x in range(len(weights_vector)):
                weights_vector[x] = weights_vector[x] / sum
        pos_weights_per_club[club] = weights_vector
    np_array = np.empty((0,len(pos_essential_ft)))


    for club in all_clubs:
        n_row = np.array(pos_weights_per_club[club])
        n_row.shape = (1,len(pos_essential_ft))
        np_array = np.append(np_array, n_row, axis = 0)
    scaler = StandardScaler()

    np_array = scaler.fit_transform(np_array)


    count = 0
    for club in all_clubs:
        scaler = MinMaxScaler()
        n_ar = np_array[count]
        n_ar.shape = (len(pos_essential_ft),1)
        n_ar = scaler.fit_transform(n_ar)
        n_ar.shape = (len(pos_essential_ft),)
        pos_weights_per_club[club] = (n_ar/np.sum(n_ar)).tolist()
        count += 1

    return pos_weights_per_club


# ## Strikers

striker_df = F_norm_df[(F_norm_df["Best Position"] == "ST") | (F_norm_df["Best Position"] == "CF") | (F_norm_df["Best Position"] == "LF")]
striker_df  =  striker_df.reset_index(drop = True)

ST_detailed_features = ['Weak Foot', 'Skill Moves', 'Height', 'Weight', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']


X = striker_df[ST_detailed_features].to_numpy()
y = striker_df["Ovr_pot"].to_numpy()
fs = SelectKBest(score_func=f_regression, k = 10)
X_selected = fs.fit_transform(X, y)

ST_essential_ft = list(fs.get_feature_names_out(ST_detailed_features))

ST_essential_ft

# ## Wingers

wingers_df = F_norm_df[(F_norm_df["Best Position"] == "LW") | (F_norm_df["Best Position"] == "RW")]
wingers_df  =  wingers_df.reset_index(drop = True)

Wing_detailed_features = ['Weak Foot', 'Skill Moves', 'Height', 'Weight', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']

X = wingers_df[Wing_detailed_features].to_numpy()
y = wingers_df["Ovr_pot"].to_numpy()
fs = SelectKBest(score_func=f_regression, k = 10)
X_selected = fs.fit_transform(X, y)

Wing_essential_ft = list(fs.get_feature_names_out(Wing_detailed_features))

# ## Midfield

mids_df = F_norm_df[(F_norm_df["Best Position"] == "RM") | (F_norm_df["Best Position"] == "CM") | (F_norm_df["Best Position"] == "CDM") | (F_norm_df["Best Position"] == "LM")]
mids_df  =  mids_df.reset_index(drop = True)

Mid_detailed_features = ['Weak Foot', 'Skill Moves', 'Height', 'Weight', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']

X = mids_df[Mid_detailed_features].to_numpy()
y = mids_df["Ovr_pot"].to_numpy()
fs = SelectKBest(score_func=f_regression, k = 10)
X_selected = fs.fit_transform(X, y)

Mid_essential_ft = list(fs.get_feature_names_out(Mid_detailed_features))

# ## CAM

cam_df = F_norm_df[(F_norm_df["Best Position"] == "CAM")]
cam_df  =  cam_df.reset_index(drop = True)

Cam_detailed_features = ['Weak Foot', 'Skill Moves', 'Height', 'Weight', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']

X = cam_df[Cam_detailed_features].to_numpy()
y = cam_df["Ovr_pot"].to_numpy()
fs = SelectKBest(score_func=f_regression, k = 10)
X_selected = fs.fit_transform(X, y)

Cam_essential_ft = list(fs.get_feature_names_out(Cam_detailed_features))

# ## Defenders

centrebacks_df = F_norm_df[(F_norm_df["Best Position"] == "CB")]
centrebacks_df  = centrebacks_df.reset_index(drop = True)

Def_detailed_features = ['Weak Foot', 'Skill Moves', 'Height', 'Weight', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']

X = centrebacks_df[Def_detailed_features].to_numpy()
y = centrebacks_df["Ovr_pot"].to_numpy()
fs = SelectKBest(score_func=f_regression, k = 10)
X_selected = fs.fit_transform(X, y)

Def_essential_ft = list(fs.get_feature_names_out(Def_detailed_features))

# ## Fullbacks

fullbacks_df = F_norm_df[(F_norm_df["Best Position"] == "LB") | (F_norm_df["Best Position"] == "RB") | (F_norm_df["Best Position"] == "LWB") | (F_norm_df["Best Position"] == "RWB")]
fullbacks_df  = fullbacks_df.reset_index(drop = True)

Def_detailed_features = ['Weak Foot', 'Skill Moves', 'Height', 'Weight', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']

X = fullbacks_df[Def_detailed_features].to_numpy()
y = fullbacks_df["Ovr_pot"].to_numpy()
fs = SelectKBest(score_func=f_regression, k = 10)
X_selected = fs.fit_transform(X, y)

Fullback_essential_ft = list(fs.get_feature_names_out(Def_detailed_features))

# ## Goalkeeper

gk_df = F_norm_df[(F_norm_df["Best Position"] == "GK")]
gk_df  = gk_df.reset_index(drop = True)

Gk_detailed_features = ['Height', 'Weight', 'ShortPassing', 'Volleys', 'Curve','Agility', 'Reactions', 'Balance',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
        'Positioning', 'Vision', 'Composure','GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']

X = gk_df[Gk_detailed_features].to_numpy()
y = gk_df["Ovr_pot"].to_numpy()
fs = SelectKBest(score_func=f_regression, k = 10)
X_selected = fs.fit_transform(X, y)

Gk_essential_ft = list(fs.get_feature_names_out(Gk_detailed_features))

all_clubs = list(F_all_df["Club"].unique())

def get_pos_weights(pos_df,pos_essential_ft):
    pos_weights_per_club={}

    for club in all_clubs:
        all_features = pos_df[pos_df["Club"] == club]
        all_features = all_features.sort_values('Ovr_pot',ascending = False).head(20)
        essential_features = all_features[pos_essential_ft]
        weights_vector=[]
        sum = 0


        for i in essential_features:
            if essential_features[i].std() != 0:
                weights_vector.append(1/essential_features[i].std())
                sum += 1/essential_features[i].std()
            else:
                weights_vector.append(0)
                sum += 0
        if(sum != 0):
            for x in range(len(weights_vector)):
                weights_vector[x] = weights_vector[x] / sum
        pos_weights_per_club[club] = weights_vector
    np_array = np.empty((0,len(pos_essential_ft)))


    for club in all_clubs:
        n_row = np.array(pos_weights_per_club[club])
        n_row.shape = (1,len(pos_essential_ft))
        np_array = np.append(np_array, n_row, axis = 0)
    scaler = StandardScaler()

    np_array = scaler.fit_transform(np_array)


    count = 0
    for club in all_clubs:
        scaler = MinMaxScaler()
        n_ar = np_array[count]
        n_ar.shape = (len(pos_essential_ft),1)
        n_ar = scaler.fit_transform(n_ar)
        n_ar.shape = (len(pos_essential_ft),)
        pos_weights_per_club[club] = (n_ar/np.sum(n_ar)).tolist()
        count += 1

    return pos_weights_per_club

pos_essential_features = [ST_essential_ft,Wing_essential_ft,Mid_essential_ft,Cam_essential_ft,Def_essential_ft,Fullback_essential_ft]

scaler = sklearn.preprocessing.MinMaxScaler()
F_std_df = F_all_df.copy(deep = True)
F_std_df[detailed_features] = scaler.fit_transform(F_std_df[detailed_features].to_numpy())

df_22_std = F_std_df.tail(df_22.shape[0])
df_21_std = F_std_df.tail(df_21.shape[0])
df_20_std = F_std_df.tail(df_20.shape[0])
df_19_std = F_std_df.tail(df_19.shape[0])
df_18_std = F_std_df.tail(df_18.shape[0])
df_17_std = F_std_df.tail(df_17.shape[0])

striker_df = F_std_df[(F_std_df["Best Position"] == "ST") | (F_std_df["Best Position"] == "CF") | (F_std_df["Best Position"] == "LF")]
striker_df = striker_df.reset_index(drop = True)

wingers_df = F_std_df[(F_std_df["Best Position"] == "LW") | (F_std_df["Best Position"] == "RW")]
wingers_df = wingers_df.reset_index(drop = True)

mids_df = F_std_df[(F_std_df["Best Position"] == "RM") | (F_std_df["Best Position"] == "CM") | (F_std_df["Best Position"] == "CDM") | (F_norm_df["Best Position"] == "LM")]
mids_df = mids_df.reset_index(drop = True)

cam_df = F_std_df[(F_std_df["Best Position"] == "CAM")]
cam_df = cam_df.reset_index(drop = True)

centrebacks_df = F_std_df[(F_std_df["Best Position"] == "CB")]
centrebacks_df = centrebacks_df.reset_index(drop = True)

fullbacks_df = F_std_df[(F_std_df["Best Position"] == "LB") | (F_std_df["Best Position"] == "RB") | (F_std_df["Best Position"] == "LWB") | (F_std_df["Best Position"] == "RWB")]
fullbacks_df = fullbacks_df.reset_index(drop = True)

gk_df = F_std_df[(F_std_df["Best Position"] == "GK")]
gk_df = gk_df.reset_index(drop = True)

striker_weights_per_club = get_pos_weights(striker_df,ST_essential_ft)
wing_weights_per_club = get_pos_weights(wingers_df,Wing_essential_ft)
mid_weights_per_club = get_pos_weights(mids_df,Mid_essential_ft)
cam_weights_per_club = get_pos_weights(cam_df,Cam_essential_ft)
def_weights_per_club = get_pos_weights(centrebacks_df,Def_essential_ft)
fullback_weights_per_club = get_pos_weights(fullbacks_df,Fullback_essential_ft)
gk_weights_per_club = get_pos_weights(gk_df,Gk_essential_ft)

# According to the player's overall rating give them weights and come up with a weighted average
# for each position in the club.


def get_cur_pos_philosophy(cur_season_df, club , pos_list,pos_essential_ft):
    club_df = cur_season_df[cur_season_df["Club"]==club]
    pos_df = club_df[club_df['Best Position'].isin(pos_list)]
    essential_df = pos_df[pos_essential_ft]
    cur_players = np.array(essential_df)
    denom = 0
    num = np.zeros(shape=(cur_players.shape[1]))

    for i in range(cur_players.shape[0]):
        overall_rating = pos_df.iloc[i]["Ovr_pot"]
        if(overall_rating >= 85):
            w = 20
        elif (overall_rating >= 80):
            w = 16
        elif(overall_rating >= 75):
            w = 8
        elif(overall_rating >= 70):
            w = 4
        else:
            w = 1
        denom += w
        num += w * cur_players[i]
    return num/denom


# similarity through weighted PCC

def club_pos_sim(new_player,club,pos_list,pos_essential_features,pos_weights_per_club, means, club_cur = None):

    new_player = np.array(new_player[pos_essential_features])
    new_player.shape = [new_player.shape[0],1]
    new_player = normalize(new_player, axis=0)
    new_player.shape = [new_player.shape[0],]

    club_cur = get_cur_pos_philosophy(F_std_df,club,pos_list,pos_essential_features)
    club_cur.shape = [club_cur.shape[0],1]
    club_cur = normalize(club_cur, axis = 0)
    club_cur.shape = [club_cur.shape[0],]

    weighted_dot = np.sum(pos_weights_per_club * (new_player-means) * (club_cur-means))
    mag_1 = np.sqrt(np.dot(pos_weights_per_club, np.square(new_player - means)))
    mag_2 = np.sqrt(np.dot(pos_weights_per_club, np.square(club_cur - means)))
    pos_sim = weighted_dot/(mag_1 * mag_2)
    return pos_sim

def get_sim(ID,club_rec,df):
    pos_df = df
    new_player = pos_df[(pos_df["ID"]==ID)]
    pos_list = list(new_player["Best Position"])
    if (pos_list == ["ST"] or ["CF"] or ["LF"]):
        pos_df= df[df['Best Position'].isin(pos_list)]
        new_player = pos_df[(pos_df["ID"]==ID)]
        essential_df = pos_df[ST_essential_ft]
        pos_weights_per_club = striker_weights_per_club
        pos_essential_features = ST_essential_ft

    if (pos_list == ["LW"] or ["RW"]):
        pos_df= df[df['Best Position'].isin(pos_list)]
        new_player = pos_df[(pos_df["ID"]==ID)]
        essential_df = pos_df[Wing_essential_ft]
        pos_weights_per_club = wing_weights_per_club
        pos_essential_features = Wing_essential_ft

    if (pos_list == ["RM"] or ["CM"] or ["CDM"] or ["LM"]):
        pos_df= df[df['Best Position'].isin(pos_list)]
        new_player = pos_df[(pos_df["ID"]==ID)]
        essential_df = pos_df[Mid_essential_ft]
        pos_weights_per_club = mid_weights_per_club
        pos_essential_features = Mid_essential_ft

    if (pos_list == ["CAM"]):
        pos_df= df[df['Best Position'].isin(pos_list)]
        new_player = pos_df[(pos_df["ID"]==ID)]
        essential_df = pos_df[Cam_essential_ft]
        pos_weights_per_club = cam_weights_per_club
        pos_essential_features = Cam_essential_ft

    if (pos_list == ["CB"]):
        pos_df = df[df['Best Position'].isin(pos_list)]
        new_player = pos_df[(pos_df["ID"]==ID)]
        essential_df = pos_df[Def_essential_ft]
        pos_weights_per_club = def_weights_per_club
        pos_essential_features = Def_essential_ft

    if (pos_list == ["LB"] or ["LWB"] or ["RB"] or ["RWB"]):
        pos_df= df[df['Best Position'].isin(pos_list)]
        new_player = pos_df[(pos_df["ID"]==ID)]
        essential_df = pos_df[Fullback_essential_ft]
        pos_weights_per_club = fullback_weights_per_club
        pos_essential_features = Fullback_essential_ft

    if (pos_list == ["GK"]):
        pos_df= df[df['Best Position'].isin(pos_list)]
        new_player = pos_df[(pos_df["ID"]==ID)]
        essential_df = pos_df[Gk_essential_ft]
        pos_weights_per_club = gk_weights_per_club
        pos_essential_features = Gk_essential_ft

    pos_np = essential_df.to_numpy()
    pos_np = normalize(pos_np)
    means = np.mean(pos_np, axis = 0)
    new_player = new_player.squeeze()
    club = new_player["Club"]
    t = club_pos_sim(new_player,club,pos_list,pos_essential_features,pos_weights_per_club[club_rec], means, club_cur= None)
    return(t)


def sim_func(ID,club_rec):
    L = [df_22_std,df_21_std,df_20_std,df_19_std,df_18_std,df_17_std]
    for i in range(6):
        pos_df= L[i]
        new_player = pos_df[(pos_df["ID"] == ID)]

        if(new_player.empty):
            pass
        else:
            t = get_sim(ID,club_rec,L[i])
            break
    else:
        t = 0
    return t


print(sim_func(231942, "Manchester City"))
