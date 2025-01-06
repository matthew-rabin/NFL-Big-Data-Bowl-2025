# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#--------------------------------PREP DATA---------------------------

#plays data

plays = pd.read_csv('C:\\Users\\rabinm\\Documents\\Portfolio\\Big Data Bowl\\Big Data Bowl 2025\\data\\nfl-big-data-bowl-2025\\plays.csv')
    #limit to sample game
    #plays = plays[plays['gameId'] == 2022102302]
    #drop columns that are not needed. limit to pass plays
pass_plays = plays[['gameId', 'playId', 'possessionTeam', 'quarter', 'gameClock', 'absoluteYardlineNumber', 'yardsToGo', 'down',
               'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation', 'receiverAlignment', 'passLocationType',
               'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability']]
pass_plays = pass_plays[plays['passLocationType'].notnull()]
    #create play_result (outcome variable). drop passLocationType column.
pass_plays['pass_play'] = 1 
pass_plays = pass_plays.drop('passLocationType', axis=1) 
                                                                
    #drop columns that are not needed. limit to rush plays
rush_plays = plays[['gameId', 'playId', 'possessionTeam', 'quarter', 'gameClock', 'absoluteYardlineNumber', 'yardsToGo', 'down',
               'preSnapHomeScore', 'preSnapVisitorScore', 'offenseFormation','receiverAlignment', 'rushLocationType',
               'preSnapHomeTeamWinProbability', 'preSnapVisitorTeamWinProbability']]
rush_plays = rush_plays[plays['rushLocationType'].notnull()]
    #create play_result (outcome variable). drop rushLocationType column.
rush_plays['pass_play'] = 0
rush_plays = rush_plays.drop('rushLocationType', axis=1) 
        
#merge pass and rush plays back to one dataframe
plays = pd.concat([pass_plays,rush_plays])

#convert quarter, minutes, seconds into game_time_remaining_seconds
plays[['minutes', 'seconds']] = plays['gameClock'].str.split(':', expand=True)

plays['minutes_remaining'] = 60-(15*plays['quarter']) + plays['minutes'].apply(int)
plays['game_time_remaining_seconds'] = plays['minutes_remaining']*60 + plays['seconds'].apply(int)

#identify home team 1/0

#games data
games = pd.read_csv('C:\\Users\\rabinm\\Documents\\Portfolio\\Big Data Bowl\\Big Data Bowl 2025\\data\\nfl-big-data-bowl-2025\\games.csv')
    #limit to sample game
    #games = games[games['gameId'] == 2022102302]
    #drop columns that are not needed. 
games = games[['gameId', 'homeTeamAbbr']]

    #merge game and play data to add home team. flag if possessionTeam = home team
plays = pd.merge(plays,games,how='left',left_on=['gameId'],right_on=['gameId'])
plays['home_team'] = (plays['possessionTeam'] == plays['homeTeamAbbr']).astype(float)
    #drop columns that are not needed. 
plays = plays.drop(['quarter', 'gameClock', 'minutes', 'seconds', 'minutes_remaining', 'homeTeamAbbr'], axis=1) 

    #modify 'preSnapHomeScore', 'preSnapVisitorScore' to score_diff. add preSnapWinProbability
home_team_plays = plays[plays['home_team'] == 1]
home_team_plays['score_diff'] = home_team_plays['preSnapHomeScore'] - home_team_plays['preSnapVisitorScore']
home_team_plays['preSnapWinProbability'] = home_team_plays['preSnapHomeTeamWinProbability']

away_team_plays = plays[plays['home_team'] == 0]
away_team_plays['score_diff'] = away_team_plays['preSnapVisitorScore'] - away_team_plays['preSnapHomeScore']
away_team_plays['preSnapWinProbability'] = away_team_plays['preSnapVisitorTeamWinProbability']

plays = pd.concat([home_team_plays,away_team_plays])

    #scale the home field advantage relative to time remaining
plays['home_team_advantage']  = plays['home_team'] * plays['game_time_remaining_seconds']

    #assign dummies for categorical variables with One-Hot Encoding
offenseFormation_encoded = pd.get_dummies(plays['offenseFormation'], dtype = int)
receiverAlignment_encoded = pd.get_dummies(plays['receiverAlignment'], dtype = int)
team_encoded = pd.get_dummies(plays['possessionTeam'], dtype = int)

plays = pd.concat([plays, offenseFormation_encoded, receiverAlignment_encoded, team_encoded], axis=1)

#remove dataframes that are no longer needed
del pass_plays, rush_plays, games, offenseFormation_encoded, receiverAlignment_encoded, home_team_plays, away_team_plays

    #generate CSV
    #plays.to_csv('C:\\Users\\rabinm\\Documents\\Portfolio\\Big Data Bowl\\Big Data Bowl 2025\\output\\plays_2022102302.csv')


#_______________MODEL DATA_________

    #array of features
X = plays[['absoluteYardlineNumber'
           ,	'yardsToGo'
           ,	'down'
           , 'game_time_remaining_seconds'
           , 'home_team_advantage'
           , 'preSnapWinProbability'
           , 'score_diff'
           , 'EMPTY', 'I_FORM', 'JUMBO', 'PISTOL',	'SHOTGUN', 'SINGLEBACK',	'WILDCAT'
           , '1x0', '1x1', '2x0', '2x1', '2x2', '3x0', '3x1', '3x2', '3x3',	'4x1', '4x2'
           ,'ARI',	'ATL',	'BAL',	'BUF',	'CAR',	'CHI',	'CIN',	'CLE',	'DAL',	'DEN',	'DET',	'GB',	'HOU',	'IND',	'JAX',	
           'KC',	'LA',	'LAC',	'LV',	'MIA',	'MIN',	'NE',	'NO',	'NYG',	'NYJ',	'PHI',	'PIT',	'SEA',	'SF',	'TB',	'TEN',	'WAS'    
           ]].values

y = plays['pass_play']
    #split into train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

    #Evaluating classification models
models = {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier(),'Decision Tree': DecisionTreeClassifier()}
results = []

    #Visualizing results
for model in models.values():
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()
    
    #Test set performance
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print("{} Test Set Accuracy: {}".format(name, test_score))
    #logistic regression model has the best results
    