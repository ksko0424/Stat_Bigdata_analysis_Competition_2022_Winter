# 지하철 혼잡도 및 미세먼지 관계 분석

> 전북대학교 통계학과 빅데이터 분석 경진대회 2022 Winter — **최우수상**  
> 서울 지하철 혼잡도 예측과 역사 내 미세먼지(PM10) 원인 분석 프로젝트

[Competition](https://www.kaggle.com/c/statjbnu1) · Python/Jupyter · EDA · Machine Learning · Statistical Hypothesis Testing

---

## 1. Project Summary

이 프로젝트는 서울 지하철 데이터를 활용해 두 가지 질문을 분석했습니다.

1. **혼잡도 예측**  
   조사일자, 호선, 역명, 상·하선 구분, 역 주변 교통시설 여부 등을 활용해 하루 평균 혼잡도를 예측할 수 있는지 확인했습니다.

2. **미세먼지 원인 분석**  
   역사 내부 미세먼지 농도가 승객 혼잡도와 관련되는지, 또는 열차 운행 빈도와 같은 다른 요인이 더 큰 영향을 주는지 검토했습니다.

핵심 분석 흐름은 관측 패턴에서 출발해 가설을 세우고, 역별 데이터와 통계 검정을 통해 설명 요인을 비교하는 방식입니다.

![Analysis Flow](assets/figures/fig01_analysis_flow.png)

---

## 2. Dataset

데이터는 대회 제공 CSV 3개와 운행 빈도 보조 자료를 사용했습니다.

| File | Description |
|---|---|
| `data/raw/statjbnu1/data1.csv` | 서울교통공사 지하철 혼잡도 정보. 조사일자, 호선, 역번호, 역명, 상·하선 구분, 30분 단위 혼잡도 포함 |
| `data/raw/statjbnu1/data2.csv` | 서울 지하철 역사 대기 정보. PM10, CO2, HCHO, CO 포함 |
| `data/raw/statjbnu1/data3.csv` | 자치구별 지하철역 정보 |
| `new_data4.csv` | 노선별 운행 횟수 및 수송 인원 요약 보조 자료 |

자료 출처:

- <https://www.data.go.kr/data/15071311/fileData.do>
- <https://data.seoul.go.kr/dataList/OA-2732/F/1/datasetView.do>
- <https://www.gimi9.com/dataset/www-data-go-kr-data-filedata-15081868>

---

## 3. Key Findings

| Topic | Finding |
|---|---|
| 혼잡도 패턴 | 평일 평균 혼잡도는 출근·퇴근 시간대에 뚜렷한 peak를 보임 |
| 역사 PM10 분포 | PM10은 노선별·역별 편차가 존재함 |
| 혼잡도와 PM10 | 역 단위 평균 혼잡도만으로 PM10 차이를 설명하기에는 한계가 있음 |
| 대안 요인 | 열차 운행 빈도는 역사 내 공기질과 함께 검토할 수 있는 주요 노선 단위 요인임 |
| 통계 검정 | 원 프로젝트 요약 기준, 운행 빈도 차이가 있는 집단은 p-value < 0.05, 혼잡도만 차이나는 집단은 p-value = 0.11 |

---

## 4. Part A — 혼잡도 예측 모델링

### 4.1 변수 구성

- 전체 후보 변수
  - 조사일자: 평일/토요일/일요일
  - 호선
  - 역 번호
  - 역명
  - 구분: 상선/하선/내선/외선
  - 시간대별 혼잡도
- 최종 선택 변수
  - 조사일자
  - 호선
  - 역명
  - 역 주변 버스터미널 여부
- 예측 대상
  - 하루 평균 혼잡도

선행연구인 *기계학습을 이용한 서울 지하철 승하차 인원 예측*에서 역 주변 버스터미널 여부가 승하차 수요와 혼잡도에 영향을 줄 수 있다는 내용을 참고해, 해당 정보를 파생변수로 구성했습니다.

### 4.2 시간대별 혼잡도 패턴

![Average Subway Congestion by Time](assets/figures/fig02_congestion_hourly_profile.png)

### 4.3 모델 비교

| Model | MSE | Note |
|---|---:|---|
| Linear Regression | 72.105 | 기준 모델 |
| Random Forest | 68.459 | 더 낮은 MSE로 최종 채택 |

![Model MSE Comparison](assets/figures/fig06_model_mse_comparison.png)

---

## 5. Part B — 혼잡도와 미세먼지(PM10)의 관계 분석

### 5.1 노선별 역사 PM10 분포

대회 제공 역사 대기 정보(`data2.csv`)를 사용해 노선별 PM10 분포를 비교했습니다.

![PM10 by Line](assets/figures/fig03_pm10_by_line_distribution.png)

### 5.2 역 단위 평균 혼잡도와 PM10 비교

`data1.csv`의 시간대별 혼잡도를 역 단위 평균 혼잡도로 요약한 뒤, `data2.csv`의 역사별 PM10과 병합했습니다.

![PM10 vs Congestion](assets/figures/fig04_pm10_vs_congestion_scatter.png)

이 비교는 PM10 차이가 단순히 평균 혼잡도 하나만으로 설명되기 어렵다는 점을 확인하기 위한 탐색적 분석입니다.

### 5.3 대안 요인: 열차 운행 빈도

노선별 평일 운행 횟수와 노선 평균 PM10을 함께 비교했습니다.

![Line Frequency and PM10](assets/figures/fig05_line_frequency_pm10_overview.png)

열차 운행 빈도는 노선 단위 변수이기 때문에 역 단위 PM10과 직접적인 인과관계를 의미하지는 않습니다. 다만 혼잡도 외에도 노선 운행 구조를 함께 고려해야 함을 보여주는 보조 분석으로 사용했습니다.

### 5.4 t-test 기반 유의성 검정

원 프로젝트에서는 운행 빈도 차이와 혼잡도 차이를 분리한 집단을 비교하고 t-test를 수행했습니다.

| 비교 조건 | t-test 결과 | 해석 |
|---|---|---|
| 운행 빈도 차이가 있는 집단 | p-value < 0.05 | PM10 차이가 통계적으로 유의함 |
| 혼잡도만 차이가 있는 집단 | p-value = 0.11 | 통계적으로 유의하지 않음 |

![Reported t-test Summary](assets/figures/fig07_reported_ttest_summary.png)

---

## 6. Reproducibility

새 시각화는 아래 스크립트로 재생성할 수 있습니다.

```bash
python scripts/visualize_subway_pm10.py
```

생성 파일:

```text
assets/figures/
├── fig01_analysis_flow.png
├── fig02_congestion_hourly_profile.png
├── fig03_pm10_by_line_distribution.png
├── fig04_pm10_vs_congestion_scatter.png
├── fig05_line_frequency_pm10_overview.png
├── fig06_model_mse_comparison.png
├── fig07_reported_ttest_summary.png
└── derived_station_pm10_congestion_frequency.csv
```

---

## 7. Repository Structure

```text
.
├── data/raw/statjbnu1/
│   ├── data1.csv
│   ├── data2.csv
│   └── data3.csv
├── assets/figures/
│   ├── fig01_analysis_flow.png
│   ├── fig02_congestion_hourly_profile.png
│   ├── fig03_pm10_by_line_distribution.png
│   ├── fig04_pm10_vs_congestion_scatter.png
│   ├── fig05_line_frequency_pm10_overview.png
│   ├── fig06_model_mse_comparison.png
│   └── fig07_reported_ttest_summary.png
├── scripts/visualize_subway_pm10.py
├── FINAL.ipynb
├── train.ipynb
├── train_ML.ipynb
├── train_ML_V2.ipynb
├── train_Vis.ipynb
├── visuality.ipynb
├── new_data4.csv
├── 최종본.pdf
└── 최종본.pptx
```
