from connect import load_data
from clean_data import filtered
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error


df, X, y = load_data()
df, X, y = filtered(df, X, y)

features = ['avg_over25', 'avg_under25', 'avg_draw', 'avg_away_win', 'avg_home_win', 'avg_AHH',
    'avg_AHA', 'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target']
target = 'total_goals'

X = df[features]
y = df[target]
df['Date'] = pd.to_datetime(df['Date'])
train_df = df[(df['Date'].dt.year == 2020) | (df['Date'].dt.year == 2019)] # Use | for OR condition
test_df = df[df['Date'].dt.year == 2021]
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Establish Poisson regression model
poisson_model = PoissonRegressor()

# Training model
poisson_model.fit(X_train, y_train)

# Predictive testing set
y_pred = poisson_model.predict(X_test)

# Evaluation Model
mse = mean_squared_error(y_test, y_pred)
r2 = poisson_model.score(X_test, y_test)
print(f"(MSE): {mse}")
print(f"(R^2): {r2}")
print(y_train.describe())
print(y_test.describe())

