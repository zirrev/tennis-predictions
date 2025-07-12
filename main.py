#%% 
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score

file_path = 'data/atp_tennis.csv'

#%% 
df = pd.read_csv(file_path)

#%% 
# Filter for grass court matches only
grass_matches = df[df['Surface'] == 'Grass'].copy()

#%% 
def calculate_games_won(score):

    if pd.isna(score) or score == '':
        return 0, 0
    
    # Split by spaces to get individual sets
    sets = score.strip().split()
    player1_games = 0
    player2_games = 0
    
    for set_score in sets:
        # Split by dash to get games in each set
        games = set_score.split('-')
        if len(games) == 2:
            try:
                p1_set_games = int(games[0])
                p2_set_games = int(games[1])
                player1_games += p1_set_games
                player2_games += p2_set_games
            except ValueError:
                # Handle cases like 7-6 where it might be a tiebreak
                continue
    
    return player1_games, player2_games

# Apply the function to create new columns for grass matches only
grass_matches['Player1_Games'] = [calculate_games_won(score)[0] for score in grass_matches['Score']]
grass_matches['Player2_Games'] = [calculate_games_won(score)[1] for score in grass_matches['Score']]

#%% 
# Data Transformations

# Convert date to datetime
grass_matches['Date'] = pd.to_datetime(grass_matches['Date'])

# Create categorical encodings
grass_matches['tournament_code'] = grass_matches['Tournament'].astype('category').cat.codes
grass_matches['series_code'] = grass_matches['Series'].astype('category').cat.codes
grass_matches['court_code'] = grass_matches['Court'].astype('category').cat.codes
grass_matches['round_code'] = grass_matches['Round'].astype('category').cat.codes

# Extract time-based features
grass_matches['year'] = grass_matches['Date'].dt.year
grass_matches['month'] = grass_matches['Date'].dt.month
grass_matches['day_of_week'] = grass_matches['Date'].dt.dayofweek

# Handle ranking data - convert to numeric and handle missing values
grass_matches['Rank_1'] = pd.to_numeric(grass_matches['Rank_1'], errors='coerce').fillna(999)
grass_matches['Rank_2'] = pd.to_numeric(grass_matches['Rank_2'], errors='coerce').fillna(999)

# Create ranking difference feature
grass_matches['rank_diff'] = grass_matches['Rank_2'] - grass_matches['Rank_1']

# Create binary target variable (1 if Player_1 wins, 0 if Player_2 wins)
grass_matches['target'] = (grass_matches['Winner'] == grass_matches['Player_1']).astype(int)

# Create games difference feature
grass_matches['games_diff'] = grass_matches['Player1_Games'] - grass_matches['Player2_Games']



#%%
rf = RandomForestClassifier(n_estimators=100, random_state=42)
train = grass_matches[grass_matches["Date"] < '2024-06-24']
test = grass_matches[grass_matches["Date"] >= '2024-06-24']
# %%
predictors = [
    'Rank_1', 'Rank_2', 'rank_diff',
    'tournament_code', 'series_code', 'court_code', 'round_code',
    'year', 'month', 'day_of_week',
    'Player1_Games', 'Player2_Games', 'games_diff'
]
# %%
# Train the model with transformed data
rf.fit(train[predictors], train['target'])

# Make predictions on test set
predictions = rf.predict(test[predictors])

# Calculate accuracy
accuracy = accuracy_score(test['target'], predictions)
# %%
precision = precision_score(test['target'], predictions)
precision
# %%
