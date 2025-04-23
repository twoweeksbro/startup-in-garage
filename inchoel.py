import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
df = pd.read_csv('ames.csv')
df.columns
df.isnull().sum()

# 결측치 발견
df['GarageArea'].isnull().sum()
df['GarageYrBlt'].isnull().sum()
df['GarageCars'].isnull().sum()

df['GarageType'].isnull().sum()
df['GarageFinish'].isnull().sum()
df['GarageQual'].isnull().sum()
df['GarageCond'].isnull().sum()


df['GarageArea'].unique()
df['GarageYrBlt'].unique()
df['GarageCars'].unique()
df['GarageType'].unique()

df['GarageFinish'].unique()
df['GarageQual'].unique()
df['GarageCond'].unique()

garagefinish_mapping = {
    'Unf': 1,
    'Rfn': 2,
    'Fin': 3
}
df['GarageFinish_Num'] = df['GarageFinish'].map(garagefinish_mapping)

garagequal_mapping = {
    'TA': 3,
    'Fa': 2,
    'Gd': 4,
    'Po': 1,
    'Ex': 5 
}
df['GarageQual_Num'] = df['GarageQual'].map(garagequal_mapping)

garagecond_mapping = {
    'TA': 3,
    'Fa': 2,
    'Gd': 4,
    'Po': 1,
    'Ex': 5 
}
df['GarageCond_Num'] = df['GarageCond'].map(garagecond_mapping)





# 결측치 제거
df = df.dropna(subset=['GarageType', 'GarageFinish','GarageQual','GarageCond'])
df.shape


# 이상치 탐지
# area
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['GarageArea'])
plt.title('Box Plot for Garage Area')
plt.show()
df.columns


# 상관관계 히트맵
# saleprice, garagecar, garagearea
numeric_columns = ['SalePrice', 'GarageCars', 'GarageArea']
corr_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm',  # 색상 반전: 빨강(양의 상관), 파랑(음의 상관 또는 0)
            fmt='.2f', 
            linewidths=0.5, 
            vmin=-1, vmax=1)  # 색상 고정 범위
plt.title('Correlation Heatmap of Garage and Sale Price Variables')
plt.show()
# 오직 Garage
numeric_columns = ['GarageYrBlt', 'GarageCars', 
                   'GarageArea','GarageFinish_Num','GarageQual_Num','GarageCond_Num']
corr_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm',  # 색상 반전: 빨강(+) 파랑(-)
            fmt='.2f', 
            linewidths=0.5, 
            vmin=-1, vmax=1)  # 색상 범위 고정
plt.title('Correlation Heatmap of Garage and Sale Price Variables')
plt.show()

# 이상치 제거
# GarageArea IQR의해 이상치 제거된 이유 
Q1 = df['GarageArea'].quantile(0.25)
Q3 = df['GarageArea'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df['GarageArea'].shape   # 기존 2450
df[df['GarageArea'] > upper_bound].shape[0] # upper_bound 이상 데이터 56개

df['SalePrice'].mean() # 이상치 제거전 평균 saleprice는 181804
df[(df['GarageArea'] > Q1) & (df['GarageArea'] < Q3)]['SalePrice'].mean()
# 1사분위 3사분위내에서 saleprice는 177024

df[df['GarageArea'] > upper_bound]['SalePrice'].mean()  
# upper_bound 이상 데이터의 saleprice는 303685으로 상당히 비싸다
df = df[df['GarageArea'] < upper_bound]  # 따라서 upper_bound에 해당되는 데이터 제거
df.shape

# GarageYrBlt의 연도별 갯수 계산
yearly_counts = df['GarageYrBlt'].value_counts().sort_index()
yearly_counts = df['GarageYrBlt'].value_counts().sort_index().reset_index()
yearly_counts.columns = ['Year Built', 'Count']
fig = px.line(yearly_counts, 
              x='Year Built', 
              y='Count', 
              title='Distribution of Garage Built Year', 
              markers=True)

fig.update_layout(
    xaxis_title='Year Built',
    yaxis_title='Count',
    template='plotly_white',
    width=900,
    height=500
)
fig.show()


# Yearbuilt와 GarageArea 관계
yearly_avg_area = df.groupby('GarageYrBlt')['GarageArea'].mean()
yearly_avg_area_df = yearly_avg_area.reset_index()
yearly_avg_area_df.columns = ['GarageYrBlt', 'AvgGarageArea']
fig = px.line(
    yearly_avg_area_df,
    x='GarageYrBlt',
    y='AvgGarageArea',
    title='Average Garage Area by Built Year',
    markers=True
)
fig.update_traces(line=dict(color='firebrick'))  # 선 색깔 설정
fig.update_layout(
    xaxis_title='Built Year (GarageYrBlt)',
    yaxis_title='Average Garage Area (sqft)',
    template='plotly_white',
    width=900,
    height=500
)

fig.show()




# Yearbuilt와 GarageCars의 관계 (평균)
yearly_avg = df.groupby('GarageYrBlt')['GarageCars'].mean()
yearly_avg_df = yearly_avg.reset_index()
yearly_avg_df.columns = ['GarageYrBlt', 'AvgGarageCars']
fig = px.line(
    yearly_avg_df,
    x='GarageYrBlt',
    y='AvgGarageCars',
    title='Average Number of Garage Cars by Built Year',
    markers=True
)

fig.update_traces(line=dict(color='green'))  # 선 색깔 설정
fig.update_layout(
    xaxis_title='Built Year (GarageYrBlt)',
    yaxis_title='Average Garage Cars',
    template='plotly_white',
    width=900,
    height=500
)
fig.show()

# GarageFinish 갯수 (고정)
garagefinish_counts = df['GarageFinish'].value_counts().reset_index()
garagefinish_counts.columns = ['GarageFinish', 'Count']
color_map = {
    'Unf': '#636EFA',
    'RFn': '#EF553B',
    'Fin': '#00CC96'
}
fig = px.pie(garagefinish_counts, 
             names='GarageFinish', 
             values='Count', 
             hole=0.4, 
             title='Distribution of Garage Finish Types')
fig.update_layout(
    showlegend=True,
    width=700,
    height=500
)
fig.show()



# GarageType 갯수
garage_counts = df['GarageType'].value_counts().reset_index()
garage_counts.columns = ['GarageType', 'Count']
fig = px.pie(
    garage_counts,
    names='GarageType',
    values='Count',
    hole=0.4,
    title='Distribution of Garage Types (Donut Chart)'
)
fig.update_traces(
    textinfo='percent+label',
    marker=dict(line=dict(color='white', width=2))
)
fig.update_layout(
    showlegend=True,
    width=700,
    height=500
)
fig.show()

# GarageQual 갯수
garage_counts = df['GarageType'].value_counts().reset_index()
garage_counts.columns = ['GarageType', 'Count']
fig = px.pie(
    garage_counts,
    names='GarageType',
    values='Count',
    hole=0.4,
    title='Distribution of Garage Types (Donut Chart)'
)
fig.update_traces(marker=dict(line=dict(color='white', width=2)))
fig.update_layout(
    showlegend=True,
    width=700,
    height=500
)
fig.show()

# GarageCond 갯수
garagecond_counts = df['GarageCond'].value_counts().reset_index()
garagecond_counts.columns = ['GarageCond', 'Count']
fig = px.pie(
    garagecond_counts,
    names='GarageCond',
    values='Count',
    hole=0.4,
    title='Distribution of Garage Condition (Donut Chart)'
)
fig.update_traces(
    textinfo='percent+label',
    marker=dict(line=dict(color='white', width=2)),
    rotation=90 
)
fig.update_layout(
    showlegend=True,
    width=700,
    height=500
)
fig.show()
