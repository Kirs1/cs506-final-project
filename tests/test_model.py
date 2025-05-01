import pytest
import pandas as pd
from xgboost import XGBClassifier
from connect import load_data
from clean_data import filtered

@pytest.fixture
def small_data():
    df, X, y = load_data()
    df, X, y = filtered(df, X, y)
    df = df.head(100)
    X = df[['avg_over25', 'avg_under25', 'avg_draw']]
    y = (df['total_goals'] > 2).astype(int)
    return X, y

def test_xgb_fits_without_error(small_data):
    X, y = small_data
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
