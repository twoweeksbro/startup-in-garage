import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
df = pd.read_csv('ames.csv')
df.columns
df.isnull().sum()
df.shape
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
# 이미지 파일 경로
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
garage_counts = df['GarageQual'].value_counts().reset_index()
garage_counts.columns = ['GarageQual', 'Count']
fig = px.pie(
    garage_counts,
    names='GarageQual',
    values='Count',
    hole=0.4,
    title='Distribution of GarageQual (Donut Chart)'
)
fig.update_traces(textinfo='percent+label',
    marker=dict(line=dict(color='white', width=2)),
                  rotation=90)
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

#################

import pandas as pd
import plotly.graph_objects as go
from palmerpenguins import load_penguins
# 데이터 로딩 및 전처리
af = load_penguins().dropna()
# total_count = len(af)
# 단계별 필터링
# step1 = af
# step2 = step1[step1['body_mass_g'] > 3000]
# step3 = step2[step2['flipper_length_mm'] > 190]
# step4 = step3[step3['bill_length_mm'] > 45]
df['GarageFinish_Num'].unique()

total_count = len(df)
step1 = df
step2 = step1[step1['GarageQual_Num'] >= 3]
step3 = step2[step2['GarageFinish_Num'] >= 2]
step4 = step3[step3['GarageArea'] > 576]
step4.shape
# 각 단계별 개체 수
step_counts = [len(step1), len(step2), len(step3), len(step4)]
# y축 라벨 (단계 + 조건 설명)
step_labels = [
    "1단계: 전체 차고지 수",
    "2단계: 차고지 퀄리티 평균 이상",
    "3단계: 차고지 마감 상태 평균 이상",
    "4단계: 차고지 면적 576이상"
]
# 퍼널 내부 텍스트 (마리 수 + 전체 비율)
text_labels = [
    f"{count}개 ({count / total_count * 100:.1f}%)"
    for count in step_counts
]

# 막대 색상 리스트 (단계별 색상)
colors =["#A3C9F1", "#D1A6E0", "#A4E6B2", "#4C6A92"]

# 퍼널 그래프 생성
fig = go.Figure(go.Funnel(
    y=step_labels,
    x=step_counts,
    text=text_labels,
    textinfo="text",
    textposition="inside",
    textfont=dict(size=16, family="Arial", weight="bold"),
    marker=dict(color=colors)  # 색상 적용
))

# 레이아웃 설정
fig.update_layout(
    title={
        "text": "단계별 조건을 만족하는 차고지 개체 수",
        "x": 0.5,
        "xanchor": "center",
        "font": dict(size=24)
    },
    paper_bgcolor='white',   # 전체 배경 흰색
    plot_bgcolor='white',     # 플롯 영역 배경 흰색
    margin=dict(t=80, l=100, r=50, b=80),
    font=dict(size=16, family="Arial")
)

fig.show()