from connect import load_data
from clean_data import filtered
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    df, X, y = load_data()
    df, X, y = filtered(df, X, y)

    df['FTR_code'] = np.where(
        df['home_goals'] > df['away_goals'], 0,
        np.where(df['home_goals'] == df['away_goals'], 1, 2)
    )

    features = [
        'avg_over25', 'avg_under25', 'avg_draw',
        'avg_away_win', 'avg_home_win',
        'avg_AHH', 'avg_AHA',
        'home_shots', 'away_shots',
        'home_shots_on_target', 'away_shots_on_target'
    ]

    X = df[features]
    y = df['FTR_code']

    df['Date'] = pd.to_datetime(df['Date'])
    train = df[df['Date'].dt.year.isin([2019, 2020])]
    test  = df[df['Date'].dt.year == 2021]

    X_train, y_train = train[features], train['FTR_code']
    X_test,  y_test  = test[features],  test['FTR_code']

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Home Win','Draw','Away Win']
    ))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['H','D','A']
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix: Match Result Prediction")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    importances = model.feature_importances_

    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
    plt.title("Feature Importance: Match Result Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    overall_acc = accuracy_score(y_test, y_pred)
    acc_home  = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])
    acc_draw  = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
    acc_away  = accuracy_score(y_test[y_test == 2], y_pred[y_test == 2])

    labels = ['Home Win', 'Draw', 'Away Win', 'Overall']
    values = [acc_home, acc_draw, acc_away, overall_acc]

    plt.figure(figsize=(8, 5))
    plt.plot(labels, values, marker='o', linewidth=2)


    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.ylim(0, 1.05)
    plt.ylabel('Accuracy / Recall')
    plt.title('Class-wise Recall & Overall Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()