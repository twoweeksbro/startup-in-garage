import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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
    'RFn': 2,
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
# yearly_counts = df['GarageYrBlt'].value_counts().sort_index()
# plt.figure(figsize=(12, 6))
# plt.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-', color='b')
# plt.title('Distribution of Garage Built Year')
# plt.xlabel('Year Built')
# plt.ylabel('Count')
# plt.grid(True)
# plt.show()
yearly_counts = df['GarageYrBlt'].value_counts().sort_index()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=yearly_counts.index,
    y=yearly_counts.values,
    mode='lines+markers',
    line=dict(color='royalblue'),
    marker=dict(size=6),
    name='Garage Count'
))
fig.update_layout(
    title='Distribution of Garage Built Year',
    xaxis_title='Year Built',
    yaxis_title='Count',
    template='plotly_white',
    width=900,
    height=500
)
fig.show()



# Yearbuilt와 GarageArea 관계
# plt.figure(figsize=(10, 6))
# plt.scatter(df['GarageYrBlt'], df['GarageArea'], alpha=0.6)
# plt.title('Garage Area vs Built Year')
# plt.xlabel('Built Year (GarageYrBlt)')
# plt.ylabel('Garage Area (sqft)')
# plt.show()
# # 선그래프로 나타냈을때
# yearly_avg_area = df.groupby('GarageYrBlt')['GarageArea'].mean()
# plt.figure(figsize=(12, 6))
# plt.plot(yearly_avg_area.index, yearly_avg_area.values, marker='o', linestyle='-', color='r')
# plt.title('Average Garage Area by Built Year')
# plt.xlabel('Built Year (GarageYrBlt)')
# plt.ylabel('Average Garage Area (sqft)')
# plt.grid(True)
# plt.show()
# plotly로 나타냈을때
yearly_avg_area = df.groupby('GarageYrBlt')['GarageArea'].mean()
# Plotly 선그래프 생성
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=yearly_avg_area.index,
    y=yearly_avg_area.values,
    mode='lines+markers',
    line=dict(color='firebrick'),
    marker=dict(size=6),
    name='Avg Garage Area'
))
fig.update_layout(
    title='Average Garage Area by Built Year',
    xaxis_title='Built Year (GarageYrBlt)',
    yaxis_title='Average Garage Area (sqft)',
    template='plotly_white',
    width=900,
    height=500
)
fig.show()





# Yearbuilt와 GarageCars의 관계 (평균)
# yearly_avg = df.groupby('GarageYrBlt')['GarageCars'].mean()
# plt.figure(figsize=(12, 6))
# plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linestyle='-', color = 'g')
# plt.title('Average Number of Garage Cars by Built Year')
# plt.xlabel('Built Year (GarageYrBlt)')
# plt.ylabel('Average Garage Cars')
# plt.grid(True)
# plt.show()
# plotly 그래프
yearly_avg = df.groupby('GarageYrBlt')['GarageCars'].mean()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=yearly_avg.index,
    y=yearly_avg.values,
    mode='lines+markers',
    line=dict(color='green'),
    marker=dict(size=6),
    name='Avg Garage Cars'
))
fig.update_layout(
    title='Average Number of Garage Cars by Built Year',
    xaxis_title='Built Year (GarageYrBlt)',
    yaxis_title='Average Garage Cars',
    template='plotly_white',
    width=900,
    height=500
)
fig.show()


# GarageFinish 갯수 (고정)
# garagefinish_counts = df['GarageFinish'].value_counts()
# plt.figure(figsize=(8, 8))
# plt.pie(garagefinish_counts, labels=garagefinish_counts.index, autopct='%1.1f%%', startangle=140)
# plt.title('Distribution of Garage Types')
# plt.axis('equal')  # 원형 유지
# plt.show()
# plotly 그래프
garagefinish_counts = df['GarageFinish'].value_counts()
fig = go.Figure(data=[go.Pie(
    labels=['Unf(마감되지 않는 상태)', 'Rfn(부분적으로 마감된)', 'Fin(마감된)'],
    values=garagefinish_counts.values,
    hole=0.4,  # 도넛 형태
    textinfo='percent+label',
    marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA']),
)])
fig.update_layout(
    title='Distribution of Garage Finish Types',
    showlegend=True,
    width=700,
    height=500
)
fig.show()



# GarageType 갯수
# # 도넛 차트
# garage_counts = df['GarageType'].value_counts()
# plt.figure(figsize=(8, 8))
# plt.pie(garage_counts, 
#         labels=garage_counts.index, 
#         autopct='%1.1f%%', 
#         startangle=140,
#         wedgeprops={'width': 0.4})  # 중심 비워서 도넛처럼 만듦
# plt.title('Distribution of Garage Types (Donut Chart)')
# plt.axis('equal')  # 원형 유지
# plt.show()
# plotly 그래프
garage_counts = df['GarageType'].value_counts()
fig = go.Figure(data=[go.Pie(
    labels=garage_counts.index,
    values=garage_counts.values,
    hole=0.4,  # 도넛 차트
    textinfo='percent+label',
    marker=dict(line=dict(color='white', width=2))
)])
fig.update_layout(
    title='Distribution of Garage Types (Donut Chart)',
    showlegend=True,
    width=700,
    height=500
)
fig.show()


# GarageQual 갯수
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='GarageQual', order=df['GarageQual'].value_counts().index)
plt.title('Distribution of Garage Quality')
plt.xlabel('Garage Quality')
plt.ylabel('Count')
plt.show()
# 도넛차트
# garagequal_counts = df['GarageQual'].value_counts()
# plt.figure(figsize=(8, 8))
# plt.pie(garagequal_counts, 
#         labels=garagequal_counts.index, 
#         autopct='%1.1f%%', 
#         startangle=140,
#         wedgeprops={'width': 0.4})  # 도넛형태로
# plt.title('Distribution of Garage Quality (Donut Chart)')
# plt.axis('equal')  # 원형 유지
# plt.show()
# plotly 그래프
garagequal_counts = df['GarageQual'].value_counts()
fig = go.Figure(data=[go.Pie(
    labels=garagequal_counts.index,
    values=garagequal_counts.values,
    hole=0.4,  # 도넛 형태
    textinfo='percent+label',
    marker=dict(line=dict(color='white', width=2)),
    rotation=90
)])
fig.update_layout(
    title='Distribution of Garage Quality (Donut Chart)',
    showlegend=True,
    width=700,
    height=500
)
fig.show()

# GarageCond 갯수
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='GarageCond', order=df['GarageCond'].value_counts().index)
plt.title('Distribution of Garage Condition')
plt.xlabel('Garage Condition')
plt.ylabel('Count')
plt.show()
# # 도넛차트
# garagecond_counts = df['GarageCond'].value_counts()
# plt.figure(figsize=(8, 8))
# plt.pie(garagecond_counts, 
#         labels=garagecond_counts.index, 
#         autopct='%1.1f%%', 
#         startangle=140,
#         wedgeprops={'width': 0.4})  # 도넛 형태
# plt.title('Distribution of Garage Condition (Donut Chart)')
# plt.axis('equal')  # 원형 유지
# plt.show()
garagecond_counts = df['GarageCond'].value_counts()
fig = go.Figure(data=[go.Pie(
    labels=garagecond_counts.index,
    values=garagecond_counts.values,
    hole=0.4,  # 도넛 형태
    textinfo='percent+label',
    marker=dict(line=dict(color='white', width=2)),
     rotation=90
)])
fig.update_layout(
    title='Distribution of Garage Condition (Donut Chart)',
    showlegend=True,
    width=700,
    height=500
)
fig.show()



