# startup-in-garage
startup-in-garage
## 팀원 소개

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/yewon-kim-alt">
        <img src="https://github.com/yewon-kim-alt.png" width="100px;" alt="yewon-kim-alt"/>
        <br />
        <sub><b>김예원</b></sub>
      </a>
      <br />
      역할1<br />
      역할2<br />
      역할3
    </td>
    <td align="center">
      <a href="https://github.com/P-fe">
        <img src="https://github.com/P-fe.png" width="100px;" alt="P-fe"/>
        <br />
        <sub><b>신인철</b></sub>
      </a>
      <br />
      역할1<br />
      역할2<br />
      역할3
    </td>
    <td align="center">
      <a href="https://github.com/twoweeksbro">
        <img src="https://github.com/twoweeksbro.png" width="100px;" alt="twoweeksbro"/>
        <br />
        <sub><b>이주형</b></sub>
      </a>
      <br />
      역할1<br />
      역할2<br />
      역할3
    </td>
    <td align="center">
      <a href="https://github.com/ui-ju">
        <img src="https://github.com/ui-ju.png" width="100px;" alt="ui-ju"/>
        <br />
        <sub><b>정의주</b></sub>
      </a>
      <br />
      역할1<br />
      역할2</strong><br />
      역할3
    </td>
  </tr>
</table>




## 프로젝트 요구사항

**활용 데이터**: Ames 데이터(lon, lat 버전)

1. 데이터 탐색(EDA) 및 전처리 결과 시각화

   - 주요 변수 분포, 결측치 처리, 이상치 탐지 등

2. 지도 기반 시각화
   - 예: Folium, Plotly 등 사용 가능
3. 인터랙티브 요소

   - 예: Plotly 등

4. 모델 학습 페이지
   - 회귀모델 훈련 과정과 결과 시각화
   - 페널티 회귀 모델 필수 사용
5. 스토리텔링 구성
   - 전체 대시보드가 하나의 분석 흐름으로 자연스럽게 이어질 것
   - 꼭 집값 예측이 아니어도 됨!
6. 전체 분량
   - 4-5페이지로 구성

<br />

# 프로젝트 개요

## 프로젝트 명
**"차고 창업에 적합한 부동산 인사이트 대시보드"**

<br>

## 프로젝트 배경
차고(Garage)는 단순히 차량을 보관하는 공간을 넘어 최근에는 개인 창고, 소규모 창업 공간(목공, 정비, 전자조립 등) 또는 스타트업의 첫 사무실로 활용되는 등 다양한 비즈니스 수요가 증가하고 있다.

Ames Housing 데이터셋은 주택의 다양한 요소를 포함하며 차고(Garage) 관련 정보도 담겨 있다. 이 데이터를 바탕으로 차고 창업에 적합한 주택을 탐색할 수 있는 인사이트를 제공하고자 한다.

<br>

## 프로젝트 목표
- Ames Housing 데이터를 분석하여 차고 관련 변수와 주택 가치/특성 간의 관계를 파악한다.

- 다양한 조건을 반영해 차고 창업에 적합한 주택의 특징을 도출한다.

- 그 결과를 직관적으로 보여주는 대시보드를 구축한다.

- 사용자가 차고지 조건을 설정하면 창업 적합 주택을 필터링 및 추천할 수 있도록 한다.

<br>

## 세부 분석 항목
### 1. 차고 관련 변수 탐색


- `GarageType`: 차고의 유형 (Detached, Attached, BuiltIn 등)

- `GarageCars`: 차고 차량 수 (공간 크기 대체 가능)

- `GarageArea`: 차고 면적 (창업 가능 최소 공간 설정 가능)

- `GarageFinish`: 내부 마감 수준 (창업 용도 적합도 판단)

- `GarageYrBlt`: 차고 건축년도 (상대적 노후도 평가)

- `GarageQual`: 차고의 품질

- `GarageCond`: 차고의 상태

시각화 예시: 히트맵, 상관계수 플롯, boxplot(차고 상태별 주택 가격)

### 2. 창업에 적합한 차고 기준 정의
- 창업에 적합한 최소 면적 (GarageArea ≥ 특정값)

- 마감이 되어 있을 것 (GarageFinish != 'Unf')

- 최근에 지어진 차고 (GarageYrBlt ≥ 2000)

- 차고 품질 및 상태가 평균 이상 (GarageQual, GarageCond ≥ 'TA')

참고: 대시보드에서 필터로 조건 설정 가능

### 3. 주택 가치와의 관계 분석
- 차고 조건에 따른 SalePrice 분포 분석

- GarageArea와 SalePrice 간 회귀 분석

- 창업 적합 주택의 지역 분포 (Neighborhood) 파악

시각화 예시: 지역별 평균 가격, Garage 특성 시각화

### 4. 추천 로직 또는 필터링 기능
- 사용자가 필터 입력:
  - 차고 면적 ≥ 400 sqft
  - `GarageFinish`: `RFn` or `Fin`
  - 차고 품질 `Gd` 이상
- 조건에 부합하는 주택을 표로 추천

- 필터 조건에 따라 대시보드 실시간 업데이트


<br>


## 대시보드 기능 요약

| 기능         | 설명 |
|--------------|------|
| 필터링 기능   | 차고 면적, 상태, 형태 |
| 시각화       | 차고 변수별 주택 가격 분포 / 회귀 분석 그래프 |
| 추천 리스트   | 창업 적합 주택 리스트 (표 형태 또는 지도 형태) |
| 지도 시각화(선택) | 지역별 창업 가능 주택 표시 지도 |
