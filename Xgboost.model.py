from connect import load_data
from clean_data import filtered
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score


df, X, y = load_data()
df, X, y = filtered(df, X, y)

df['goal_group'] = pd.cut(
    df['total_goals'],
    bins=[-1, 2, 100],     
    labels=[0, 1]
).astype(int)

features = [
    'avg_over25', 'avg_under25', 'avg_draw', 'avg_away_win', 'avg_home_win',
    'avg_AHH', 'avg_AHA', 'home_shots', 'away_shots',
    'home_shots_on_target', 'away_shots_on_target'
]
X = df[features]
y = df['goal_group']

df['Date'] = pd.to_datetime(df['Date'])
train_df = df[(df['Date'].dt.year == 2020) | (df['Date'].dt.year == 2019)] # Use | for OR condition
test_df = df[df['Date'].dt.year == 2021]
X_train = train_df[features]
y_train = train_df['goal_group']
X_test = test_df[features]
y_test = test_df['goal_group']

model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1 Score: {f1}")
print(classification_report(y_test, y_pred))
counts = df['goal_group'].value_counts()
print(counts)