import pandas as pd

ames_df = pd.read_csv('week10/startup-in-garage/data/ames.csv')

# ames
# GarageType, GarageYrBlt, GarageFinish, 
# GarageCars, GarageArea, GarageQual, GarageCond

garage_cols = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
                'GarageArea', 'GarageQual', 'GarageCond']

ames_df[garage_cols].info()

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




# folium 버전

ames_df = ames_df.dropna(subset=garage_cols)



# 이상치 제거
# GarageArea IQR의해 이상치 제거된 이유 
Q1 = ames_df['GarageArea'].quantile(0.25)
Q3 = ames_df['GarageArea'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
ames_df['GarageArea'].shape   # 기존 2450
ames_df[ames_df['GarageArea'] > upper_bound].shape[0] # upper_bound 이상 데이터 56개

ames_df['SalePrice'].mean() # 이상치 제거전 평균 saleprice는 181804
ames_df[(ames_df['GarageArea'] > Q1) & (ames_df['GarageArea'] < Q3)]['SalePrice'].mean()
# 1사분위 3사분위내에서 saleprice는 177024

ames_df[ames_df['GarageArea'] > upper_bound]['SalePrice'].mean()  
# upper_bound 이상 데이터의 saleprice는 303685으로 상당히 비싸다
ames_df = ames_df[ames_df['GarageArea'] < upper_bound]  # 따라서 upper_bound에 해당되는 데이터 제거







import folium
import pandas as pd
from IPython.display import display

# 중심 좌표
center_lat, center_lon = 42.02601, -93.63975

# folium 지도 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=12.5, tiles='CartoDB positron',
               width=900,height=700)

# 컬러 매핑 함수
def color_scale(value, max_val):
    scale = int(255 * value / max_val)
    return f'#{scale:02x}{scale:02x}ff'  # 파란색 계열

# 최대 GarageArea 계산
max_area = ames_df['GarageArea'].max()

# 마커 추가
for _, row in ames_df.iterrows():
    lat, lon, area = row['Latitude'], row['Longitude'], row['GarageArea']
    
    if pd.notnull(lat) and pd.notnull(lon) and pd.notnull(area):
        folium.CircleMarker(
            location=[lat, lon],
            radius=area / 50,
            color=color_scale(area, max_area),
            fill=True,
            fill_color=color_scale(area, max_area),
            fill_opacity=0.6,
            popup=f"GarageArea: {area} sqft<br>SalePrice: ${row['SalePrice']:,.0f}"
        ).add_to(m)

# ▶ 범례 HTML 삽입
legend_html = '''
<div style="
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 9999;
    background-color: white;
    padding: 10px 15px;
    border: 2px solid #ccc;
    border-radius: 5px;
    font-size: 14px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
">
    <b>Garage Area Legend</b><br>
    <i style="background: #0000ff; width: 12px; height: 12px; display: inline-block;"></i> Small<br>
    <i style="background: #7f7fff; width: 12px; height: 12px; display: inline-block;"></i> Medium<br>
    <i style="background: #ffff; width: 12px; height: 12px; display: inline-block;"></i> Large
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# ▶ 타이틀 삽입 (HTML)
title_html = '''
<h3 align="center" style="font-size:20px"><b>Ames Housing - Garage Area Map</b></h3>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 저장 및 표시
m.save('garage_map.html')
display(m)

import os



import folium
from folium.plugins import MarkerCluster
import pandas as pd

# 중심 좌표
center_lat, center_lon = 42.02601, -93.63975

# folium 지도 생성
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12.5,
    tiles='cartodbpositron',
    width=900,
    height=700
)

# 마커 클러스터 추가
marker_cluster = MarkerCluster().add_to(m)

# 컬러 매핑 함수 (작을수록 밝은 파랑, 클수록 진한 파랑)
def color_scale(value, max_val):
    scale = int(255 * value / max_val)
    return f'#{scale:02x}{scale:02x}ff'

# 최대 GarageArea 계산
max_area = ames_df['GarageArea'].max()

# 마커 반복 추가
for _, row in ames_df.iterrows():
    lat, lon, area = row['Latitude'], row['Longitude'], row['GarageArea']

    if pd.notnull(lat) and pd.notnull(lon) and pd.notnull(area):
        folium.CircleMarker(
            location=[lat, lon],
            radius=area / 50,
            color=color_scale(area, max_area),
            fill=True,
            fill_color=color_scale(area, max_area),
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>GarageArea:</b> {area} sqft<br><b>SalePrice:</b> ${row['SalePrice']:,.0f}",
                max_width=300
            )
        ).add_to(marker_cluster)

# ▶ 범례 HTML 삽입
legend_html = '''
<div style="
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 9999;
    background-color: white;
    padding: 10px 15px;
    border: 2px solid #ccc;
    border-radius: 5px;
    font-size: 14px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
">
    <b>Garage Area Legend</b><br>
    <i style="background: #d0d0ff; width: 12px; height: 12px; display: inline-block;"></i> Small<br>
    <i style="background: #7f7fff; width: 12px; height: 12px; display: inline-block;"></i> Medium<br>
    <i style="background: #0000ff; width: 12px; height: 12px; display: inline-block;"></i> Large
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# ▶ 타이틀 삽입 (HTML)
title_html = '''
<h3 align="center" style="font-size:20px"><b>Ames Housing - Garage Area Map</b></h3>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 저장 및 렌더링
m.save('garage_map.html')
