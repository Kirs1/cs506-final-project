# cs506-final-project
Project Objectives：Predicting the number of goals in a soccer match based on odds

Project Description： The odds themselves are set by bookmakers based on a variety of factors (including team strength, betting volume, etc.), reflecting the market's expectations of the outcome of the game. By analyzing the changes in the odds, we can also obtain prediction information on the number of goals in the game.

Collecting data：
1.Odds data:The initial odds and real-time odds offered by bookmakers for the game, including odds of win, draw, loss, over/under, etc.
Source: Obtained by accessing the API of the odds data provider,(Such as SportOddsAPI, OddsMatrix) Some websites provide collated football datasets.(WhoScored, FootyStats and so on)

2.Historical match result data: match date, home and away teams, scores, etc.
Source: same as above.

Data modeling plan:
1.Multiple regression model:Taking odds as independent variables and the total number of goals in the game as dependent variables, a multiple regression model is established to analyze the relationship between odds and the number of goals.

2.Machine learning model: Application: Use algorithms such as random forest and XGBoost to predict the number of goals based on odds data.

Data visualization plan:In order to intuitively display the data and model prediction results, the following visualization methods are planned:
1.Scatter plot of odds and actual number of goals: evaluate the relationship between odds and number of goals.
2.Scatter plot of actual number of goals and predicted number of goals: evaluate the prediction effect of the model.

Test plan: In order to evaluate the performance of the model, the following test strategy is planned:
Dataset division: The collected data is divided into training set and test set in chronological order, for example, using the data of the 2010-2018 season(Leagues in different regions) for training and the data of the 2019-2020 season for testing.
