# 🔮 수요 예측 캡스톤 프로젝트

> 시계열 데이터를 활용한 수요 예측 모델 개발 및 성능 비교 연구

## 📋 프로젝트 개요

본 프로젝트는 시계열 데이터를 활용하여 **미래 수요를 예측**하는 다양한 모델들을 구현하고 성능을 비교 분석하는 캡스톤 연구입니다.

### 🎯 주요 목표

- **단기 예측**: t 시점에서 t+1 예측
- **다중 예측**: t 시점에서 t+1, t+2 동시 예측
- **베이스라인 구축**: 3가지 기본 예측 모델 성능 평가
- **확장 가능한 프레임워크**: 고도화된 모델 추가를 위한 기반 마련

## 📊 데이터셋

- **기간**: 1981년 1월 1일 ~ 1984년 12월 30일
- **총 데이터**: 1,460개 일별 기록
- **컬럼**: Date (날짜), Sales (매출)
- **파일**: `data/for_test.csv`

## 🛠 구현된 모듈

### 1. 성능 평가 모듈 (`performance.py`)

시계열 예측 모델의 성능을 평가하는 통합 도구입니다.

```python
from performance import evaluate_forecast

# 간단 사용법
results = evaluate_forecast(actual, predicted)
print(results)  # {'wmape': 21.04, 'mape': 23.45, ...}
```

**제공 지표**:

- **WMAPE** (기본): Weighted Mean Absolute Percentage Error
- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MASE**: Mean Absolute Scaled Error

### 2. 베이스라인 모델 (`models/baseline_test.py`)

3가지 기본 예측 모델을 구현하여 성능 기준점을 제공합니다.

```python
from models.baseline_test import BaselineModels

baseline_tester = BaselineModels(data)
results = baseline_tester.evaluate_all_baselines(horizon=1)
baseline_tester.plot_single_prediction_point(horizon=1)
```

**구현된 모델**:

1. **전일 베이스라인** 🔴

   - 전날 값을 그대로 사용
   - 가장 단순하지만 효과적인 방법

2. **전주 베이스라인** 🟠

   - 1주일 전 같은 요일 값 사용
   - 주간 패턴이 있는 데이터에 효과적

3. **전년 베이스라인** 🟣
   - 1년 전 같은 날 값 사용
   - 계절성 패턴 반영, 윤년 고려

## 📈 성능 결과

### 베이스라인 모델 성능 비교 (WMAPE %)

| 모델                | t+1 예측   | t+1,t+2 예측 |
| ------------------- | ---------- | ------------ |
| **전일 베이스라인** | **21.04%** | **24.09%**   |
| 전주 베이스라인     | 28.36%     | 28.37%       |
| 전년 베이스라인     | 30.04%     | 30.06%       |

> 🏆 **결론**: 전일 베이스라인이 두 예측 시나리오 모두에서 최고 성능 달성

## 🎨 시각화 기능

### 집중형 예측 시각화

- **최근 1주일 컨텍스트**: 예측 배경 정보 제공
- **실제 값 vs 예측 값**: 명확한 비교를 위한 점 표시
- **참조점 표시**: 각 모델이 사용하는 기준 값 표시
- **작년 데이터 주석**: 별도 텍스트 박스로 표시

### 성능 비교 차트

- **막대 그래프**: 모델 간 성능 직관적 비교
- **수치 표시**: 정확한 성능 지표 확인
- **다중 시나리오**: t+1, t+1,t+2 병렬 비교

## 🚀 사용 방법

### 1. 빠른 시작

```bash
# 베이스라인 테스트 실행
python main_test.py
```

### 2. Jupyter Notebook 사용

```python
# main_test.ipynb에서 셀 단위로 실행
# 대화형 분석 및 시각화 가능
```

### 3. 개별 모듈 활용

```python
# 성능 평가만 사용
from performance import TimeSeriesEvaluator

evaluator = TimeSeriesEvaluator()
results = evaluator.evaluate(actual, predicted, metrics=['wmape', 'mae'])
```

## 📁 프로젝트 구조

```
forecasting_archive/
├── 📄 README.md              # 프로젝트 설명서 (이 파일)
├── 📄 CLAUDE.md              # 개발 컨텍스트 및 기술 문서
├── 📄 performance.py         # 성능 평가 모듈
├── 📄 main_test.py           # 테스트 스크립트
├── 📄 main_test.ipynb        # Jupyter 노트북
├── 📂 data/
│   └── 📄 for_test.csv       # 테스트 데이터셋
└── 📂 models/
    ├── 📄 baseline.py        # 기존 베이스라인
    └── 📄 baseline_test.py   # 베이스라인 모델 구현
```

## 🔧 요구사항

```python
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
```

---

<div align="center">

</div>
