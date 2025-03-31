from connect import load_data
from clean_data import filtered
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

df, X, y = load_data()
df, X, y = filtered(df, X, y)

df['total_goals_class'] = df['total_goals'].clip(upper=9) 
features = [
        'avg_over25', 'avg_under25', 'avg_away_win', 'avg_AHH', 'avg_AHA',
        'avg_home_win', 'avg_draw', 'home_shots', 'away_shots',
        'home_shots_on_target', 'away_shots_on_target'
        ]

X = df[features]
y = df['total_goals_class'].astype(int)
df['Date'] = pd.to_datetime(df['Date'])
train_df = df[(df['Date'].dt.year == 2020) | (df['Date'].dt.year == 2019)] # Use | for OR condition
test_df = df[df['Date'].dt.year == 2021]
X_train = train_df[features]
y_train = train_df['total_goals_class']
X_test = test_df[features]
y_test = test_df['total_goals_class']

# Model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# prediction
y_pred = clf.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))