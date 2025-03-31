import matplotlib.pyplot as plt
from connect import load_data
if __name__ == "__main__":
    df, X, y = load_data()
    # Histogram: Distribution of total goals scored
    plt.hist(y, bins=range(0, 14), align='left', edgecolor='black')
    plt.xticks(range(0, 10))  
    plt.xlabel('Total Goals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Goals')
    plt.show()
    
    counts = df['total_goals'].value_counts().sort_index()
    print(counts)
    
    #Scatter chart: odds of main win vs. total goals scored
    plt.scatter(df['avg_home_win'], y)
    plt.xlabel('Market Avg Home Win Odds')
    plt.ylabel('Total Goals')
    plt.title('Home Win Odds vs Total Goals')
    plt.show()
    
    #Scatter chart: away odds vs. total goals scored
    plt.scatter(df['avg_away_win'], y)
    plt.xlabel('Market Avg Away Win Odds')
    plt.ylabel('Total Goals')
    plt.title('Away Win Odds vs Total Goals')
    plt.show()
    
    #Scatter plot: draw odds vs. total goals scored
    plt.scatter(df['avg_draw'], y)
    plt.xlabel('Market Avg Draw Odds')
    plt.ylabel('Total Goals')
    plt.title('Draw Odds vs Total Goals')
    plt.show()
    
    #Scatter plot: over2.5 odds vs. total goals scored
    plt.scatter(df['avg_over25'], y)
    plt.xlabel('Market Avg Over 2.5 Goals Odds')
    plt.ylabel('Total Goals')
    plt.title('Over 2.5 Goals Odds vs Total Goals')
    plt.show()
    
    #Scatter plot: under2.5 odds vs. total goals scored
    plt.scatter(df['avg_under25'], y)
    plt.xlabel('Market Avg Under 2.5 Goals Odds')
    plt.ylabel('Total Goals')
    plt.title('Under 2.5 Goals Odds vs Total Goals')
    plt.show()
    
    #Scatter plot: Asian Handicap for Home odds vs. total goals scored
    plt.scatter(df['avg_AHH'], y)
    plt.xlabel('Market Avg Asian Handicap Home Odds')
    plt.ylabel('Total Goals')
    plt.title('Asian Handicap Home Odds vs Total Goals')
    plt.show()
    
    #Scatter plot: Asian Handicap for Away odds vs. total goals scored
    plt.scatter(df['avg_AHA'], y)
    plt.xlabel('Market Avg Asian Handicap Away Odds')
    plt.ylabel('Total Goals')
    plt.title('Asian Handicap Away Odds vs Total Goals')
    plt.show()
    
    #Scatter plot: Home shots vs. total goals scored
    plt.scatter(df['home_shots'], y)
    plt.xlabel('Home shots')
    plt.ylabel('Total Goals')
    plt.title('Home shots vs Total Goals')
    plt.show()
    
    #Scatter plot: Away shots vs. total goals scored
    plt.scatter(df['away_shots'], y)
    plt.xlabel('Away shots')
    plt.ylabel('Total Goals')
    plt.title('Away shots vs Total Goals')
    plt.show()
    
    #Scatter plot: home_shots_on_target vs. total goals scored
    plt.scatter(df['home_shots_on_target'], y)
    plt.xlabel('Home shots on target')
    plt.ylabel('Total Goals')
    plt.title('Home shots on target vs Total Goals')
    plt.show()
    
    #Scatter plot: away_shots_on_target vs. total goals scored
    plt.scatter(df['away_shots_on_target'], y)
    plt.xlabel('Away shots on target')
    plt.ylabel('Total Goals')
    plt.title('Away shots on target vs Total Goals')
    plt.show()

def filtered(df, X, y):
    df_filtered = df[(df['total_goals'] <= 9) & (df['avg_home_win'] < 20) & (df['avg_away_win'] <= 35) & (df['avg_draw']<= 14) & (df['avg_under25'] <= 5) & (df['avg_AHH']<= 2.5) & (df['avg_AHA']>= 1.5) & (df['home_shots']<= 35) & (df['away_shots']<= 35) & (df['home_shots_on_target']<=16) & (df['away_shots_on_target']<=15)]
    features = [
        'avg_over25', 'avg_under25', 'avg_away_win', 'avg_AHH', 'avg_AHA',
        'avg_home_win', 'avg_draw', 'home_shots', 'away_shots',
        'home_shots_on_target', 'away_shots_on_target'
    ]
    target = 'total_goals'

    X_filtered = df_filtered[features]
    y_filtered = df_filtered[target]
    return df_filtered, X_filtered, y_filtered
    