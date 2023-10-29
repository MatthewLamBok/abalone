import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Normalization
def normalization(df, normalization_type):
    print(" testests set")
    sex_column = df['Sex']
    df_without_sex = df.drop('Sex', axis=1)
    if normalization_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    if normalization_type == "StandardScaler":
        scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_without_sex), columns=df_without_sex.columns)
    df_normalized['Sex'] = sex_column
    return df

def info_correlation(df, display = False):
    sex_column = df['Sex']
    df = df.drop('Sex', axis=1)
    correlation_matrix = df.corr(method='pearson')
    if display == True:
        plt.figure(figsize=(20,8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation')
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv('./data.csv')
    print(df)

    #INFOMATION
    #Check Imbalanced Data Ans: Close enough
    class_distribution = df['Sex'].value_counts()
    print(class_distribution)

    df.isna().mean().to_frame(name='% of missing values') # No missing values

    normalization_type = 'MinMaxScaler'
    normalization(df, normalization_type)
    #df = pd.get_dummies(df, columns=['Sex'])
    info_correlation(df, display = True)
    
    
    print (df)