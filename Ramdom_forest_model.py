from connect import load_data
from clean_data import filtered
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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

cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title("Random Forest Confusion Matrix: Total Goals (0-9)")
plt.xlabel("Predicted Goals")
plt.ylabel("Actual Goals")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
importance = clf.feature_importances_
sns.barplot(x=importance, y=features)
plt.title("Ramdom_forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

recalls = recall_score(y_test, y_pred, average=None, labels=list(range(10)))


overall_acc = accuracy_score(y_test, y_pred)


labels = list(range(10)) + ['Overall']
values = list(recalls) + [overall_acc]


plt.figure(figsize=(10, 6))
plt.plot(labels, values, marker='o', linewidth=2)


for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.ylim(0, 1.05)
plt.xlabel('Goal Count Class (0–9) and Overall')
plt.ylabel('Recall / Accuracy')
plt.title('Recall by True Goal Class & Overall Accuracy\n(Random Forest)')
plt.grid(True)
plt.tight_layout()
plt.show()
