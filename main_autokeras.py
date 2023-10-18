
import numpy as np
import pandas as pd


# Load data
df = pd.read_csv('./data.csv')
x = df.drop('Sex', axis=1).values
y_encoded = pd.get_dummies(df['Sex']).values

# Split data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)




