from connect import load_data
from clean_data import filtered
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df, X, y = load_data()
df, X, y = filtered(df, X, y)

counts = df['total_goals'].value_counts().sort_index()
print(counts)


df_corr = pd.concat([X, y], axis=1)
corr_data = df_corr.corr()

# Draw a heat map
plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap with Total Goals")
plt.tight_layout()
plt.show()