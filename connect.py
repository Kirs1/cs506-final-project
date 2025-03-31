import sqlite3
import pandas as pd


def load_data():
    db_path = 'database.sqlite'
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        Date,
        HomeTeam,              
        AwayTeam,              
        FTHG                  AS home_goals,
        FTAG                  AS away_goals,
        `Avg>2.5`            AS avg_over25,
        `Avg<2.5`            AS avg_under25,
        `AvgA`               AS avg_away_win,
        `AvgAHH`             AS avg_AHH,
        `AvgAHA`             AS avg_AHA,
        `AvgH`               AS avg_home_win,
        `AvgD`               AS avg_draw,
        `HS`                 AS home_shots,
        `AS`                 AS away_shots,
        `HST`             AS home_shots_on_target,
        `AST`             AS away_shots_on_target
    FROM football_data
    WHERE
        FTHG IS NOT NULL
        AND FTAG IS NOT NULL
        AND `Avg>2.5`  IS NOT NULL
        AND `Avg<2.5`  IS NOT NULL
        AND `AvgA`     IS NOT NULL
        AND `AvgAHH`   IS NOT NULL
        AND `AvgAHA`   IS NOT NULL
        AND `AvgH`     IS NOT NULL
        AND `AvgD`     IS NOT NULL
        AND `HS`       IS NOT NULL
        AND `AS`       IS NOT NULL
        AND `HST`      IS NOT NULL
        AND `AST`      IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    df['total_goals'] = df['home_goals'] + df['away_goals']


    features = [
        'avg_over25',    
        'avg_under25',   
        'avg_AHH',
        'avg_AHA',
        'avg_home_win',
        'avg_away_win',
        'avg_draw',
        'home_shots',
        'away_shots',
        'home_shots_on_target',
        'away_shots_on_target',
    ]

    X = df[features]
    y = df['total_goals']   

    return df, X, y

