import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data.csv')
df.isna().mean().to_frame(name='% of missing values') # No missing values
df = pd.get_dummies(df, columns=['Sex'])
correlation_matrix = df.corr(method='pearson')
plt.figure(figsize=(20,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation')
plt.show()