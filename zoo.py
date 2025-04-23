import pandas as pd

ames_df = pd.read_csv('week10/startup-in-garage/data/ames.csv')

# ames
# GarageType, GarageYrBlt, GarageFinish, 
# GarageCars, GarageArea, GarageQual, GarageCond

garage_cols = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
                'GarageArea', 'GarageQual', 'GarageCond']

# 결측치 확인
ames_df[garage_cols].isnull().sum()


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


# 수치형 변수 이상치 확인.
for col in num_cols:
    print(ames_df[col].describe())
    print('\n') 



# Area에 너무 큰게 있다.    
print(ames_df['GarageArea'].describe())
print(ames_df['GarageCars'].value_counts())
print(ames_df['GarageCars'].describe())

    


# 수치형 변수 상관계수
ames_df[garage_cols].select_dtypes('number').corr()


# 수치형 변수 상관계수 시각화.
sns.heatmap(ames_df[garage_cols].select_dtypes('number').corr())






# 지도 기반 시각화
import plotly.express as px

# 위도/경도 + 집값 정보만 필터링
ames_df = pd.read_csv('week10/startup-in-garage/data/ames.csv')

ames_df.isnull().sum()
# 전처리
ames_df_clean = ames_df.dropna(subset=garage_cols)
# year
ames_df_clean.isnull().sum()
# 상업지구. 무슨 업종. 양질의 인터렉티브시각화 퀄리티.
import plotly.express as px

# GarageArea
# Plotly scatter_mapbox 시각화
fig = px.scatter_mapbox(
    ames_df_clean,
    lat="Latitude",
    lon="Longitude",
    color="GarageArea",              # 색깔로 집값 표현
    size="GarageArea",               # 점 크기로도 집값 표현
    color_continuous_scale= "blues",  # 색상 팔레트
    size_max=13,
    zoom=11.5,
    height=650,
    width=800,
    mapbox_style="carto-positron",  # 깔끔한 지도 스타일
    title="Ames Housing: 위치 기반 집값 시각화"
)

fig.update_layout(

    mapbox_center={"lat": 42.02601, "lon": -93.63975},
    margin={"r":0,"t":30,"l":0,"b":0})

fig.show()


# GarageYrBlt
# Plotly scatter_mapbox 시각화
fig = px.scatter_mapbox(
    ames_df_clean,
    lat="Latitude",
    lon="Longitude",
    color="GarageYrBlt",              # 색깔로 집값 표현
    size="GarageYrBlt",               # 점 크기로도 집값 표현
    color_continuous_scale= "blues",  # 색상 팔레트
    size_max=10,
    zoom=11.5,
    height=650,
    width=800,
    mapbox_style="carto-positron",  # 깔끔한 지도 스타일
    title="Ames Housing: 위치 기반 집값 시각화"
)

fig.update_layout(

    mapbox_center={"lat": 42.02601, "lon": -93.63975},
    margin={"r":0,"t":30,"l":0,"b":0})

fig.show()

ames_df_clean

# GarageFinish
fig = px.scatter_mapbox(
    ames_df,
    lat="Latitude",
    lon="Longitude",
    color="GarageFinish",  # 범주형 변수
    hover_data=["SalePrice", "GarageArea"],  # 마우스 오버시 표시
    size_max=20,
    zoom=12,
    mapbox_style="carto-positron",
    height=700,
    width=1000,
    title="🏠 GarageFinish 별 주택 위치 시각화"
)

fig.update_layout(
    mapbox_center={"lat": 42.02601, "lon": -93.63975},
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

fig.show()


garage_cols
# GarageType
fig = px.scatter_mapbox(
    ames_df,
    lat="Latitude",
    lon="Longitude",
    color="GarageType",  # 범주형 변수
    hover_data=["SalePrice", "GarageArea"],  # 마우스 오버시 표시
    size_max=20,
    zoom=12,
    mapbox_style="carto-positron",
    height=700,
    width=1000,
    title="🏠 Garage Type 별 주택 위치 시각화"
)

fig.update_layout(
    mapbox_center={"lat": 42.02601, "lon": -93.63975},
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

fig.show()

ames_df['UniformSize'] = 3  # 원하는 고정 크기

# GarageQual
fig = px.scatter_mapbox(
    ames_df,
    lat="Latitude",
    lon="Longitude",
    color="GarageQual",  # 범주형 변수
    hover_data=["SalePrice", "GarageQual"],  # 마우스 오버시 표시
    size="UniformSize",
    size_max= 11,
    zoom=12,
    mapbox_style="carto-positron",
    height=700,
    width=1000,
    title="🏠 Garage Type 별 주택 위치 시각화"
)

fig.update_layout(
    mapbox_center={"lat": 42.02601, "lon": -93.63975},
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)

fig.show()

