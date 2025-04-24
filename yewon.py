import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 데이터 로드 및 전처리
df = pd.read_csv('ames.csv')

garagefinish_mapping = {'Unf':1, 'Rfn':2, 'Fin':3}
garagequal_mapping = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
garagecond_mapping = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}

df['GarageFinish_Num'] = df['GarageFinish'].map(garagefinish_mapping)
df['GarageQual_Num'] = df['GarageQual'].map(garagequal_mapping)
df['GarageCond_Num'] = df['GarageCond'].map(garagecond_mapping)

df = df.dropna(subset=['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'])

Q1 = df['GarageArea'].quantile(0.25)
Q3 = df['GarageArea'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df = df[df['GarageArea'] < upper_bound]

features = [
    'GarageYrBlt', 'GarageCars', 'GarageArea', 
    'GarageFinish_Num', 'GarageQual_Num', 'GarageCond_Num',
    'LotArea', 'OverallCond', 'OverallQual', 'MSSubClass'
]
cat_features = ['GarageType', 'Neighborhood', 'PavedDrive', 'Utilities', 'Street', 'Functional', 'HouseStyle']

for col in cat_features:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())


# Neighborhood 상위 10개만 남기고 나머지는 'Other'로
top_neigh = df['Neighborhood'].value_counts().index[:10]
df['Neighborhood_reduced'] = df['Neighborhood'].where(df['Neighborhood'].isin(top_neigh), 'Other')

# cat_features 변수에 Neighborhood_reduced로 교체
cat_features = ['GarageType', 'Neighborhood_reduced', 'PavedDrive', 'Utilities', 'Street', 'Functional', 'HouseStyle']

# 원-핫 인코딩
df_encoded = pd.get_dummies(df[cat_features], drop_first=True)
df_num = df[features]
df_model = pd.concat([df_num, df_encoded, df['SalePrice']], axis=1).dropna()


X = df_model.drop('SalePrice', axis=1)
y = df_model['SalePrice']

def calculate_vif(df):
    X_ = df.copy()
    X_ = X_.assign(const=1)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_.columns
    vif_data["VIF"] = [variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

# --- VIF 반복 제거 루프 ---
X_vif = X.copy()
X_vif = X_vif.astype(float)

while True:
    vif_df = calculate_vif(X_vif)
    print("VIF 상위:\n", vif_df.head(10))
    high_vif = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
    # 상수항(const)은 제거하지 않음
    high_vif = [f for f in high_vif if f != 'const']
    if not high_vif:
        print("VIF 10 초과 변수가 없어 다중공선성 문제 없음.")
        break
    # 가장 VIF가 높은 변수 하나 제거
    print(f"VIF가 높은 변수 '{high_vif[0]}'를 제거합니다.")
    X_vif = X_vif.drop(columns=[high_vif[0]])

# --- Lasso/OLS ---
X_train, X_test, y_train, y_test = train_test_split(X_vif, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)

coef = pd.Series(lasso.coef_, index=X_vif.columns)
print("선택된 변수 및 계수:")
print(coef[coef != 0])
print("테스트 R^2:", lasso.score(X_test_scaled, y_test))

selected_features = coef[coef != 0].index.tolist()
valid_features = [f for f in selected_features if f in df_model.columns]
formula = "SalePrice ~ " + "+".join([f"Q('{feature}')" for feature in valid_features])

model = ols(formula, data=df_model).fit()
print(model.summary())









########시각화#######

### 히트맵###
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,12))
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.title('설명 변수 간 상관계수 히트맵')
plt.show()


### vif 계산###
def calculate_vif(df):
    X_ = df.copy()
    X_ = X_.assign(const=1)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_.columns
    vif_data["VIF"] = [variance_inflation_factor(X_.values, i) for i in range(X_.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

X_vif = X.copy()
X_vif = X_vif.astype(float)

removed_features = []  # 제거된 변수 저장 리스트

print("=== VIF 반복 제거 과정 ===")
while True:
    vif_df = calculate_vif(X_vif)
    print(vif_df.head(10))  # 상위 10개만 출력
    high_vif = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
    high_vif = [f for f in high_vif if f != 'const']
    if not high_vif:
        print("VIF 10 초과 변수가 없어 다중공선성 문제 없음.")
        break
    feature_to_remove = high_vif[0]
    print(f"VIF가 높은 변수 '{feature_to_remove}'를 제거합니다.\n")
    removed_features.append(feature_to_remove)
    X_vif = X_vif.drop(columns=[feature_to_remove])

print("=== 제거된 변수 목록 ===")
print(removed_features)

#=== 제거된 변수 목록 ===
#['GarageType_Attchd', 'Functional_Typ']


### 라쏘 계수 막대그래프###
plt.figure(figsize=(10,8))
coef.sort_values().plot(kind='barh', color=['red' if c < 0 else 'blue' for c in coef.sort_values()])
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Lasso 회귀 계수 (수평 막대그래프)')
plt.xlabel('계수 값')
plt.show()

# 0인 변수 (제외된 변수)
excluded_vars = coef[coef == 0]
print("\n=== Lasso에서 제외된 변수 (계수 == 0) ===")
print(excluded_vars)



####ols 요약###
from statsmodels.formula.api import ols

selected_features = coef[coef != 0].index.tolist()
valid_features = [f for f in selected_features if f in df_model.columns]
formula = "SalePrice ~ " + "+".join([f"Q('{feature}')" for feature in valid_features])

model = ols(formula, data=df_model).fit()
print(model.summary())





################시각화#################


import pandas as pd
import plotly.graph_objects as go

# VIF 제거 후 사용된 변수 리스트
used_vars = list(X_vif.columns)

# 변수 그룹 예시 (분석에 사용된 변수 중 대표적으로 묶을 수 있는 그룹)
variable_groups = {
    '차고 관련 수치형': [
        'GarageYrBlt', 'GarageCars', 'GarageArea', 
        'GarageFinish_Num', 'GarageQual_Num', 'GarageCond_Num'
    ],
    '기타 수치형': [
        'LotArea', 'OverallCond', 'OverallQual', 'MSSubClass'
    ],
    '범주형 더미 변수 일부': [col for col in used_vars if col.startswith('GarageType_')][:10],  # 예시 10개만
}

# 위 그룹에 포함된 변수만 필터링하여 실제 존재하는 변수로 재조정
for key in variable_groups:
    variable_groups[key] = [v for v in variable_groups[key] if v in used_vars]

# 히트맵 그리기용 함수
def get_corr_matrix(vars_list):
    return df_model[vars_list].corr()

# 초기 변수 그룹: 첫 번째 그룹으로 설정
init_group_name = list(variable_groups.keys())[0]
init_vars = variable_groups[init_group_name]
corr_init = get_corr_matrix(init_vars)

fig = go.Figure(
    data=go.Heatmap(
        z=corr_init.values,
        x=corr_init.columns,
        y=corr_init.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation')
    )
)

# 드롭다운 버튼 구성
buttons = []
for group_name, vars_list in variable_groups.items():
    corr = get_corr_matrix(vars_list)
    buttons.append(
        dict(
            label=group_name,
            method='update',
            args=[
                {'z': [corr.values],
                 'x': [corr.columns],
                 'y': [corr.index]},
                {'title': f'상관관계 히트맵 - {group_name}'}
            ]
        )
    )

fig.update_layout(
    title=f'설명 변수 상관관계 히트맵 - {init_group_name}',
    updatemenus=[
        dict(
            buttons=buttons,
            direction='down',
            pad={'r': 10, 't': 10},
            showactive=True,
            x=0,
            y=1.15,
            xanchor='left',
            yanchor='top'
        )
    ],
    width=800,
    height=800,
    template='plotly_white'
)

fig.show()
