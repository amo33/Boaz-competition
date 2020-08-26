# 8/27 발표용 자료

## EDA
₩서정민₩
내일 해야지..

## Modeling

### 1. LightGBM
#### Feature Engineering
- 파생변수 1 : 결측값 개수
- 파생변수 2 : categorical data -> binaray data
  - 1)LabelEncoder()를 통해 numerical 변수
  - 2)OneHotEncoder()를 통해 고유값 별로 0/1의 binary 변수를 데이터로 사용
- 파생변수 3 : ind 포함한 변수들 조합한 new_ind 생성
  - 조합, 빈도를 기반으로 파생변수 생성

#### 파라미터 튜닝
```
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 10000
params = {"objective": "binary",
          "boosting_type": "gbdt", # dart도 돌려보자
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
           "max_bin": 256,
          "feature_fraction": feature_fraction,
          "verbosity": 0,
          "drop_rate": 0.1,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9
}
```
#### Stratified K-FOLD 내부 교차검증
- 데이터가 편향되어있기때문에 단순 K-FOLD 교차검증을 사용하면 성능평가가 잘 되지 않을 수 있어 Stratified K-FOLD 내부 교차검증을 수행하였다.
- 총 16번의 seed 값으로 학습을 돌려, 평균 값을 최종 예측 결과물로 사용
- 시드값이 많을 수록 랜덤 요소로 인한 분산을 줄일 수 있음



### 2. XGBOOST
~~
### 3. Entity Embedding
~~
### 4. Neural Net
~~
## Ensemble
### 1. Stacking
~~
### 2. 가중평균
~~
