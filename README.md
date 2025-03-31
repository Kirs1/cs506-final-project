# cs506-final-project midterm repo

## Goal Predictor
The tool for predicting integer and interval goals

## Summary
Goal Predictor is a tool for predicting integer goals and interval goals (less than 2.5 goals and more than 2.5 goals). It makes predictions by inputting match data. Currently, the Random Forest model is used to predict integer goals, and the Xgboost model is used to predict interval goals.

# Data
## Dataset
We are currently using a public database from Kaggle: https://www.kaggle.com/datasets/sashchernuh/european-football.
We will try to merge it with other databases to get more features (such as weather, lineup ratings, etc.) to strengthen the correlation between the input data and the number of goals scored.

## Data preprocessing
We read all the odds related features from the database：

- AvgH = Market average home win odds
- AvgD = Market average draw win odds
- AvgA = Market average away win odds
- Avg>2.5 = Market average over 2.5 goals
- Avg<2.5 = Market average under 2.5 goals
- AvgAHH = Market average Asian handicap home team odds
- AvgAHA = Market average Asian handicap away team odds

And some features that are strongly correlated with the number of goals：
- HS = Home Team Shots
- AS = Away Team Shots
- HST = Home Team Shots on Target
- AST = Away Team Shots on Target

Also the number of goals scored：
- FTHG  = Full Time Home Team Goals
- FTAG  = Full Time Away Team Goals
- Total_goals = FTHG + FTAG





