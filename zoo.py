import pandas as pd

ames_df = pd.read_csv('week10/startup-in-garage/data/ames.csv')

# ames
# GarageType, GarageYrBlt, GarageFinish, 
# GarageCars, GarageArea, GarageQual, GarageCond

garage_cols = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
                'GarageArea', 'GarageQual', 'GarageCond']

ames_df[garage_cols].isnull().sum()

ames_df['GarageYrBlt'].min()
ames_df['GarageYrBlt'].mean()
ames_df['GarageYrBlt'].max()
ames_df['GarageYrBlt'].describe()

ames_df[garage_cols].isnull()

# 차고지 관련 변수 value_counts()
for col in garage_cols:
    print(ames_df[col].value_counts())


# 시각화 
import matplotlib.pyplot as plt

import seaborn as sns
plt.hist(ames_df['GarageArea'])
plt.hist(ames_df['GarageYrBlt'])
ames_df.plot()

# 범주형 변수 막대 그래프
cate_cols = ['GarageType','GarageFinish', 'GarageQual', 'GarageCond']
for col in cate_cols:
    sns.countplot(x=col, data=ames_df)
    plt.show()

# 수치형 변수 히스토그램 

num_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars']
for col in num_cols:
    sns.histplot(ames_df[col])
    plt.show()

# 수치형 변수 상관계수
ames_df[garage_cols].select_dtypes('number').corr()

# 수치형 변수 시각화.

sns.heatmap(ames_df[garage_cols].select_dtypes('number').corr())
