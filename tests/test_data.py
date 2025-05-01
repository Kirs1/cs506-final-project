import pytest
import pandas as pd
from connect import load_data

def test_load_data_returns_dataframe():
    df, X, y = load_data()
    assert isinstance(df, pd.DataFrame)
    for col in ['Date', 'home_goals', 'away_goals']:
        assert col in df.columns

def test_filtered_removes_nulls():
    from clean_data import filtered
    df, X, y = load_data()
    df2, X2, y2 = filtered(df, X, y)
    assert df2['home_goals'].isnull().sum() == 0
    assert df2['away_goals'].isnull().sum() == 0
