from connect import load_data
from clean_data import filtered
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc



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
    num_class=2,
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


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["≤2.5 Goals", ">2.5 Goals"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues')
plt.title("XGBoost Confusion Matrix")
plt.grid(False)
plt.show()


plt.figure(figsize=(10, 6))
importance = model.feature_importances_
sns.barplot(x=importance, y=features)
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

overall_acc = accuracy_score(y_test, y_pred)

recall_le2 = recall_score(y_test, y_pred, pos_label=0)
recall_gt2 = recall_score(y_test, y_pred, pos_label=1)


labels = ['≤2.5 Goals', '>2.5 Goals', 'Overall']
values = [recall_le2, recall_gt2, overall_acc]

plt.figure(figsize=(8, 5))
plt.plot(labels, values, marker='o', linewidth=2)

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.ylim(0, 1.05)
plt.ylabel('Recall / Accuracy')
plt.title('Recall by Class & Overall Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()



