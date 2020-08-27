### 8/27 발표용 자료

# 전처리 및 EDA
> 1. 데이터 살펴보기
> 2. 불균형 클래스
> 3. 누락값 처리
> 4. 새로운 변수 생성
> 5. 중요한 피처 선택
> 6. 피처 스케일링


### 1. 데이터 살펴보기
데이터 특징
* 변수의 이름이나 설명을 알 수 없음
* 데이터의 형태만 알 수 있음
> * 이진 변수
> * 범주 값이 정수인 범주형 변수
> * 정수 또는 실수 변수
> * 값이 -1인 누락된 변수
> * 타겟 변수와 ID 변수

|role|level|count|
|-----|-------|--------|
|id|nominal|1|
|input|binary|17|
|input|inerval|10|
|input|nominal|14|
|input|ordinal|16|
|target|binary|1|

* null 값은 -1로 표현

### 2. 불균형 클래스
> target = 0 / target = 1 비율

|label|count|
|-----|-------
|0|573518|
|1|21694|

#### UnderSampling or OverSampling
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F67qja%2Fbtqx8AVDBN8%2FFs7wjgngVukhQENwabB4tK%2Fimg.png" width="70%">

- undersampling: 유용한 정보를 잃을 수 있음
- oversampling: 과적합 발생할 수 있음

> undersampling

|label|count|
|-----|-------
|0|195246|
|1|21694|

### 3. 누락값 처리

|variable|null(%)|
|-----|-------|
|ps_car_03|68.39%|
|ps_car_05|44.26%|
|ps_reg_03|17.78%|
|ps_car_14|7.25%|
|.|.|
|.|.|
|.|.|

* ps_car_03_cat, ps_car_05_cat: 제거 (누락 값 비율 매우 높음)
* 연속형 변수 -> 평균값
* 범주형 변수 -> 최빈값

### 4. 새로운 변수 생성
1) 누락 값 개수

> target = -1(결측값)인 범주형 변수와 비율 

<img src="https://user-images.githubusercontent.com/61506233/91469446-5e3a2f80-e8ce-11ea-8535-84574c70fd8c.png" width="30%"> <img src="https://user-images.githubusercontent.com/61506233/91469458-61352000-e8ce-11ea-951f-7b84854693bf.png" width="30%"> <img src="https://user-images.githubusercontent.com/61506233/91470244-6e9eda00-e8cf-11ea-90e2-1dbb879d4427.png" width="30%">

<img src="https://user-images.githubusercontent.com/61506233/91469483-6a25f180-e8ce-11ea-9494-45fb815c0bfd.png" width="30%"> <img src="https://user-images.githubusercontent.com/61506233/91469490-6d20e200-e8ce-11ea-84ac-c5c10dc7ff04.png" width="30%"> <img src="https://user-images.githubusercontent.com/61506233/91469501-701bd280-e8ce-11ea-88ac-fdb9b2f0285d.png" width="30%">


* 누락 값의 여부를 별도의 범주로 생성

2) 상관관계 분석
> 연속형 변수 사이의 상관 관계 확인

![image](https://user-images.githubusercontent.com/61506233/91469042-d48a6200-e8cd-11ea-956f-1f16f5672111.png)

* 상관 관계가 높은 변수들끼리 차원을 늘려 새로운 변수 생성

|Before|After|
|-----|------|
|109|164|

### 5. 피처 선택 및 제거
> Selecting features with a Random Forest and SelectFromModel
> 상위 50% 변수만 선택

|features|importances|
|-------|-------|
|ps_car_11_cat_te|0.021241|
|ps_car_13|0.017428|
|ps_car_12 ps_car_13|0.017303|
|.|.|
|.|.|
|.|.|

|Before|After|
|-----|------|
|164|82|

-------------------
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
- 시드값을 주어 랜덤 요소로 인한 분산을 줄임
                
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

>  - Mean Encoding이란?
>
> 0,1 또는 임의의 숫자로 카테고리 변수를 표현하는 것이 아니라, 평균값을 이용하여 카테고리 변수를 더 의미있게 표현하고자하는 방법
> 
> 하지만 변수 내의 label 분포가 극단적일 때, 즉 불균형할 때 오버피팅의 문제 발생
> -> trainset 을 이용하여 mean encoding 해야 하는데 trainset과 달리 testset의 label 분포가 균형적이라면 mean encoding한 값이 testset을 나타낸다고 할 수 없기 때문)
> 
> 이를 해결하기 위한 방법이 smoothing (label분포가 극단적일 때 치우쳐진 평균을 전체 평균에 가깝도록 스무스하게 만드는 방법)
>   
>   (모든 피처를 Mean Encoding하여 적용한 결과 오히려 정확도가 감소, 카테고리 값이 104개로 매우 많아 label 분포가 매우 불균형한 ps_cal_11_cat에만 수행한 것으로 보임))

4. 다운샘플링 : 0,1값을 가지는 training data를 같은 개수로 맞춘 것이 아니라, 1의 비율이 전체 데이터의 0.1이 되도록 다운샘플링하였다.
               기존 training data에서 0의 비율이 1보다 압도적으로 높으므로 기존의 불균형한 분포를 유지하기 위해 위와 같이 다운샘플링한 것 같다. 

5. 카테고리 피처 더미 변수로 변환

### 모델
개별 모델들의 예측 결과로 새로운 학습데이터와 테스트데이터를 생성,
최종 스태킹 모델에 학습시킨다.

지니계수로는 2*auc-1 사용 (discussion)


### 2) 가중평균
Neural net 0.2
lightgbm 0.5
xgboost 0.3

성능이 좋은 모델에 가중치를 조금 더 높게 두고 종합하여 최종 결과 도출
