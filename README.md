# cs506-final-project repo
Midterm repo presentation:https://youtu.be/2QugIp1c2kw

## Goal Predictor
The tool for predicting integer and interval goals

## Summary
Goal Predictor is a tool for predicting integer goals and interval goals (less than 2.5 goals and more than 2.5 goals). It makes predictions by inputting match data. Currently, the Random Forest model is used to predict integer goals, and the Xgboost model is used to predict interval goals.

## Run the code
1. Make sure the required toolkit and python3 are installed on your computer.
2. Git clone cs506-final-project.
3. In the cs506-final-project folder of the terminal, enter: make. (Automatically download the environment to decompress the zip and run the code.)


# Data
## Dataset
We are currently using a public database from Kaggle: https://www.kaggle.com/datasets/sashchernuh/european-football.
We will try to merge it with other databases to get more features (such as weather, lineup ratings, etc.) to strengthen the correlation between the input data and the number of goals scored.

## Data preprocessing and visualization
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

The histogram can well show the distribution of total goals:

![The histogram can well show the distribution of total goals](picture/Figure1.png)

It can be seen that most of the goals are between 0 and 6, and it is very rare to have more than 6 goals. Therefore, in the subsequent random forest prediction of integer goals, we set 6 goals as the upper limit, and the accuracy of 7, 8, and 9 goals is almost 0.

We also made a scatter plot for each feature and the total number of goals. Scatter plots can visually show the relationship between variables and help us find outliers to better clean the data:

![The scatter plot of AvgH vs total_goals](picture/Figure2.png)

For example, in this scatter plot, we can see that there are several differences in total goals and average home win odds. Therefore, we choose to filter out data with more than 9 goals and average home win odds greater than 20 to reduce the impact of outliers on the overall prediction accuracy and stability of the model.

![The scatter plot of Avg<2.5 vs total_goals](picture/Figure6.png)

Similarly, in this scatter plot of Avg < 2.5 vs total_goals, we choose to filter out data with a total goal count greater than 9 and an average under 2.5 goals less than 5.

We plot a heat map to show the correlation between different features and the total number of goals:

![The heat map](picture/Figure13.png)

From the heat map, we can observe that shots on target have a strong positive correlation with the total number of goals. In addition, shots, odds of over/under 2.5 goals, away win and draw odds also have a certain correlation with the total number of goals. From the heat map, we can observe that shots on target have a strong positive correlation with the total number of goals. In addition, shots, odds of over/under, away wins and draws also have a certain correlation with the total number of goals. The correlation between features and total goals is crucial for the prediction model. The addition of strongly correlated features is a huge improvement for the model. This is why we added the two features of shots on target and shots. Before adding these two features, the R^2 value of the poisson regression model was only a few percent, but after adding these two features, the R^2 value increased to about 0.25. The accuracy of the predictions of the other two models also increased to varying degrees. Therefore, if we want to improve the prediction ability of the model, it is essential to add strongly correlated features in the future.

## Data modeling and preliminary results.
We currently use three different models：
- Poisson_regression_model: Evaluate the contribution of each feature to Poisson model performance metrics such as R^2 and MSE. Add/remove features to determine their importance to model performance. The aforementioned shots and shots on target features significantly improve the performance of the Poisson regression model. In the future, we will also use it to determine the importance of added features (such as weather, lineup rating, etc.) to model performance.

- Ramdom_forest_model: Used to predict the number of goals scored as an integer. Due to the uncertainty of football itself, guessing the specific number of goals often appears to be less accurate. With the current features, the accuracy is only 0.259, and it is expected to be improved by adding more relevant features in the future.

- Xgboost.model: Used to predict the interval in which goals will be scored in a match. Currently two ranges are set (less than 2.5 goals or more than 2.5 goals). The advantage of intervals is that they are more predictable than specific numbers. The predicted accuracy under the current features is 0.67. Due to the fact that the number of samples between 0-2 goals is very close to the number of samples with 3 goals or more. (As can be seen in the previous histogram) The F1 Score is very close to the accurate value (the difference is less than 1%). This is a relatively reliable result in football prediction. It is also expected that adding more relevant features in the future can increase accuracy value.

The training data for the above three models are the filtered data between 2019 and 2020, and the test data are the filtered data in 2021. Although the database itself provides data between 2001 and 2021, most of the data before 2019 lack key odds data, so the data before 2019 has been filtered out. 

## Future plan:
Merge other databases to add more features (such as weather, lineup scores, etc.), and try other models to find out if there is a better fit than the current model.

## Reproducability
In order to reproduce our result, follow the below steps:

- pip install -r requirements.txt.
- Unzip soccerdatabase.zip.
- run different model files.







