# 🔮 수요 예측 캡스톤 프로젝트

> 시계열 데이터를 활용한 수요 예측 모델 개발 및 성능 비교 연구

## 📋 프로젝트 개요

본 프로젝트는 시계열 데이터를 활용하여 **미래 수요를 예측**하는 다양한 모델들을 구현하고 성능을 비교 분석하는 캡스톤 연구입니다.

### 🎯 주요 목표

- **단기 예측**: t 시점에서 t+1 예측
- **다중 예측**: t 시점에서 t+1, t+2 동시 예측
- **베이스라인 구축**: 3가지 기본 예측 모델 성능 평가
- **확장 가능한 프레임워크**: 고도화된 모델 추가를 위한 기반 마련

## 📊 데이터셋(더미 데이터셋입니다)

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

### 2. 베이스라인 테스트 모듈 (`models/baseline_test.py`)

3가지 기본 예측 모델을 구현하여 성능 기준점을 제공합니다.
=> 후에 좋은 성능을 내는 모델 1가지를 골라서 기준점으로 사용할 예정입니다.

```python
from models.baseline_test import BaselineModels

baseline_tester = BaselineModels(data)
results = baseline_tester.evaluate_all_baselines(horizon=1)
baseline_tester.plot_single_prediction_point(horizon=1)
```

**Baseline 후보 모델 3가지**:

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

### 📊 평가 방법론

- **테스트 기간**: 1982년 1월 1일 ~ 1984년 12월 30일 (1,094개 포인트)
- **평가 방식**: 각 시점에서 미래값 예측 후 실제값과 비교
- **총 예측 수**:
  - t+1: 1,094개 예측값
  - t+1,t+2: 2,188개 예측값 (1,094 × 2)
- **WMAPE 계산**: 모든 예측값의 가중 평균 절대 백분율 오차
  - 공식: `WMAPE = Σ|실제값 - 예측값| / Σ|실제값| × 100`
  - 장점: 개별 데이터포인트에 0값이 있어도 안정적으로 계산 가능

## 🔧 요구사항

```
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
```

---

<div align="center">
<img src='data/korea.jpg' width=60px>
<div>고려대학교 일반대학원 MSBA 6기</div>
</div>
