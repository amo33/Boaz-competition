# 8/27 발표용 자료

# EDA
₩서정민₩
내일 해야지..

# Modeling
## overview

|model|private|public|
|-----|-------|--------|
|single LightGBM|0.29090| 0.28536|
|single XGBOOST|0.28947|0.28395|
|Neural Network|0.28186|0.27823|
|**Final Ensemble**|**0.29196**|**0.28673**|

## Feature Engineering
- 파생변수 1 : 결측값 개수
- 파생변수 2 : categorical data -> binaray data
  - 1)LabelEncoder()를 통해 numerical 변수
  - 2)OneHotEncoder()를 통해 고유값 별로 0/1의 binary 변수를 데이터로 사용
- 파생변수 3 : ind 포함한 변수들 조합한 new_ind 생성
  - 조합, 빈도를 기반으로 파생변수 생성
  
## 1. LightGBM

<img src="https://user-images.githubusercontent.com/61506233/91308489-4d17f280-e7ea-11ea-8b5d-c07b0b2c6bc6.png" width="40%">

#### 파라미터 튜닝
```
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 10000
params = {"objective": "binary",
          "boosting_type": "gbdt", 
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

## 2. XGBOOST

<img src="https://user-images.githubusercontent.com/61506233/91308541-61f48600-e7ea-11ea-9840-4c0efbc9ed2d.png" width="40%">

#### 파라미터 튜닝
```
 params = {'eta' : 0.025,
                'gamma' : 9,
                'max_depth' : 6,
                'reg_lambda' : 1.2,
                'colsample_bytree' : 1.0,
                'min_child_weight' : 10,
                'reg_alpha' : 8,
                'scale_pos_weight' : 1.6,
                'subsample' : 0.7,
                'eval_metric' : 'auc',
                'objective' : 'binary:logistic',
                'seed' : 2017,
                'silent' : False,
                'tree_method' : 'gpu_hist'
           }
```


#### Stratified K-FOLD 내부 교차검증

<img src="https://user-images.githubusercontent.com/61506233/91283490-a15cab80-e7c5-11ea-9774-0c762ac6074b.png" width="70%">

- 데이터가 편향되어있기때문에 단순 K-FOLD 교차검증을 사용하면 성능평가가 잘 되지 않을 수 있어 Stratified K-FOLD 내부 교차검증을 수행하였다.
- 총 16번의 seed 값으로 학습을 돌려, 평균 값을 최종 예측 결과물로 사용
- 시드값이 많을 수록 랜덤 요소로 인한 분산을 줄일 수 있음
                
## 3. Entity Embedding & Neural Network

- 데이터 셋이 neural-network에 적합하지 않다는 분석도 처음에는 나왔지만, 상위권의 지원자들이 NN 모델을 만들었기 때문에 이 과정을 진행했습니다. 
- 이 nn 모델은 entity embedding 기술을 사용했습니다. 여기서 entity embedding 은 수가 많은 변수들의 집합을 다루는데 사용되는 새로운 기법입니다. 
- 이 기법은 이번 미니 프로젝트에 적합하다고 볼수 가 있는데요. car 모델을 살펴보면 104개의 카테고리가 있는데, 이런 경우에 entity embedding을 사용한다고 합니다. 
- entity embedding 은 word embedding 과 비슷한 목적을 갖고 있다고 이해할 수 있습니다. 

## 4. Ensemble
### 1) Stacking
lightgbm 모델3개와 xgboost를 스태킹하여 메타데이터 생성, 최종 메타 모델로는 LogisticRegression을 사용하였다.

### 사용한 전처리
1. calc 피처 삭제 
2. null값 많은 피처 삭제 

3. ps_cal_11_cat 피처 Mean Encoding 수행 : 

  -Mean Encoding이란?

 0,1 또는 임의의 숫자로 카테고리 변수를 표현하는 것이 아니라, 평균값을 이용하여 카테고리 변수를 더 의미있게 표현하고자하는 방법
   (모든 피처를 Mean Encoding하여 적용한 결과 오히려 정확도가 감소)

4. 다운샘플링 : 0,1값을 가지는 training data를 같은 개수로 맞춘 것이 아니라, 1의 비율이 전체 데이터의 0.1이 되도록 다운샘플링하였다.
               기존 training data에서 0의 비율이 1보다 압도적으로 높으므로 기존의 불균형한 분포를 유지하기 위해 위와 같이 다운샘플링한 것 같다. 

5. 카테고리 피처 더미 변수로 변환

### 모델
개별 모델들의 예측 결과로 새로운 학습데이터와 테스트데이터를 생성,
최종 스태킹 모델에 학습시킨다.

지니계수로는 2*auc-1 사용 (discussion)

~~
### 2) 가중평균
~~
