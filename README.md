### 데이터 전처리

#### StandardScaler()
- 데이터의 평균을 0, 분산을 1이 되도록, 표준 정규분포를 따르게 하는 스케일링
- ±1.96을 벗어나면 이상치로 판단한다.
- <code>from sklearn.preprocessing import StandardScaler</code>

#### MinMaxScaler()
- 데이터가 0~1 사이에 위치하도록 최소값은 0, 최대값을 1로 변환한다.
- 서로 다른 단위의 feature끼리 비교가 가능해진다.
- <code>from sklearn.preprocessing import MinMaxScaler</code>

#### MaxAbsScaler()
- 모든 값을 -1~1 사이에 위치하도록, 절대값의 최소값은 0, 최대값은 1이 되도록 변환한다.
- 양의 방향에 대한 단위뿐 아니라 음의 방향에 대한 단위까지 스케일링하고자 할 때 사용한다.
- <code>from sklearn.preprocessing import MaxAbsScaler</code>
---
### AI (Artificial Intelligence)
<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/0df245ff-4235-4434-a411-d2052a9ca629" width="800px">

#### Rule-base AI
- 특정 상황을 이해하는 전문가가 직접 입력값(문제)과 특징을 전달(규칙)하여 출력값(정답)을 내보내는 알고리즘이다.
- 광범위한 데이터 수집, 정리 또는 교육이 필요하지 않으므로 전문가의 기존 지식을 기반으로 비지니스 규칙을 정의하여 구현 복잡성을 줄일 수 있다.
- 의사 결정 프로세스가 명시적인 "if-then" 사전 정의 규칙에 의존하므로 높은 투명성을 제공한다.
- 본질적으로 정적이며 유연성이 없기 때문에, 사전 정의된 규칙을 수동으로 조정하여 변화나 진화하는 조건에만 적응할 수 있다.
- 사전 정의된 규칙이 명확한 지침을 제공하지 않는 모호하거나 불확실한 상황에 직면할 때 어려움을 겪을 수 있다.
- 미리 정의된 기준에 의존하면, 전문가 개인의 편견이 들어갈 수 밖에 없고, 이로 인해 미묘한 행동을 설명하지 못할 수 있으며 잠재적으로 불공평하거나 부정확한 평가로 이어질 수 있다.

#### Machine Learning AI
- 데이터를 기반으로 규칙성(패턴)을 학습하여 결과를 추론하는 알고리즘이다.
- 현실 세계의 패턴을 분석하여 개발자가 직접 코드를 작성하는 것이 어려웠으나 머신러닝을 이용해서 해결할 수 있다.

- <sub>※</sub>데이터 마이닝, 음성 인식(언어 구분), 영상 인식(이미지 판별), 자연어 처리(번역, 문맥 찾기)에서 머신러닝이 적용된다.
- <sub>※ 데이터 마이닝이란, 대규모 데이터 안에서 체계적이고 자동적으로 통계적 규칙이나 짜임을 분석하여 가치있는 정보를 빼내는 과정이다.</sub>
- 데이터의 패턴을 학습하여 이를 통해 예측 등을 수행할 수 있다.

1. 지도 학습 (Supervised Learning)
   
> 입력값(문제)과 출력값(정답)을 전달하면, 알아서 특징을 직접 추출하는 방식이다.  
> 다른 입력값(문제)과 동일한 출력값(정답)을 전달하면 새로운 특징을 알아서 추출한다.  

> 문제(Feature)와 답(Target, Label)을 주는 것이다.  
> - 분류 (코로나 양성/음성, 사기 번호/일반 번호 등 단일 값 예측)  
> - 회귀 (1년 뒤의 매출액, 내일 주식 가격 등 연속 값 예측)  

2. 비지도 학습 (Unsupervised Learning)
   
> 입력값(문제)만 전달하고 정답 없이 특징만 추출하는 학습이다.  
> 추출한 특징에 대해 AI가 출력값(정답)을 부여하여 입력값(문제)은 출력값(정답)이라는 것을 알아낸다.  

> 문제(Feature)를 주고 답은 주지 않는 것이다.  
> - 군집화 (클러스터링, 비슷한 문제를 분석하여 편을 나누어 각 편으로 모으는 것)  
> - 차원 축소 (문제의 개수를 압축(축소)하여 함축된 의미를 찾아내는 것)  

3. 강화 학습 (Reinforcement Learning)
   
> https://kr.mathworks.com/discovery/reinforcement-learning.html

#### Machine Learning의 단점
- 데이터에 의존적이다 (Garbage In, Garbage Out), 데이터가 안좋으면 결과도 안좋을 수 밖에 없다.
- 학습 데이터로 잘 만들어진 로직을 가진 모델일지라도 실제 데이터 적용 시 정확한 결과가 나오지 않을 수 있다.
- 머신러닝 학습을 통해 로직이 생기면, 나온 결과가 어떻게 이렇게 나왔는지에 대한 분석이 쉽지 않다(블랙박스).
- 데이터를 넣으면 원하는 것처럼 좋은 결과를 얻기란 쉽지 않다.

#### R vs Python

- R
  
> 개발 언어에 익숙하지 않지만 통계 분석에 능한 현업 사용자일 경우  
> 오랜 기간동안 쌓인 다양하고 많은 통계 패키지  

- Python
  
> 직관적인 문법, 객체지향 함수형 프로그래밍 언어, 다양한 라이브러리  
> 다양한 영역 (운영체제, 서버, 네트워크 등)으로 연계 및 상용화하기 좋음  

### 머신러닝과 딥러닝은 R보다는 "파이썬"을 사용하자!

---
### 분류 (Classifier)
- 대표적인 지도학습 방법 중 하나이며, 다양한 문제와 정답을 학습한 뒤 별도의 테스트에서 정답을 예측한다.
- 주어진 문제와 정답을 먼저 학습한 뒤 새로운 문제에 대한 정답을 예측하는 방식이다.
- 이진 분류 (Binary Classification)의 경우 정답은 0(음성, Negative)과 1(양성, Positive)과 같이 True, False값을 가진다.
- 다중 분류 (Multiclass Classification)는 정답이 가질 수 있는 값은 3개 이상이다.(예: 0, 1, 2, 3).

#### 피처 (Feature)
- 데이터 세트의 일반 컬럼이며, 2차원 이상의 다차원 데이터까지 통틀어 피처라고 한다.
- 타겟을 제외한 나머지 속성을 의미한다.
#### 레이블(Label), 클래스(Class), 타겟(Target), 결정(Decision)
- 지도 학습 시, 데이터의 학습을 위해 주어지는 정답을 의미한다.
- 지도 학습 중, 분류의 경우 이를 레이블 또는 클래스라고도 부른다.

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/661c184d-230a-4409-93c8-5fcf52146b03" width="600px" style="margin-left: 10px">

### 분류 예측 프로세스
<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/6861fdf6-64a5-4229-a212-a4d402f09b5f">

#### 데이터 세트 분리
**train_test_split(feature, target, test_size, random_state)**

- 학습 데이터 세트와 테스트 데이터 세트를 분리해준다.
- feature: 전체 데이터 세트 중 feature
- target: 전체 데이터 세트 중 target
- test_size: 테스트 세트의 비율 (0 ~ 1)
- random_state: 매번 동일한 결과를 원할 때, 원하는 seed(기준점)를 작성한다.

#### 모델 학습
**fit(train_feature, train_target)**
- 모델을 학습시킬 때 사용한다.
- train_feature: 훈련 데이터 세트 중 feature
- train_target: 훈련 데이터 세트 중 target

#### 평가
**accuracy_score(y_test, predict(X_test))**
- 모델이 얼마나 잘 예측했는지를 "정확도"라는 평가 지표로 평가할 때 사용한다.
- y_test: 실제 정답
- predict(X_test): 예측한 정답
---
#### 결정 트리 (Decision Tree)
- 매우 쉽고 유연하게 적용될 수 있는 알고리즘으로서 데이터의 스케일링, 정규화 등의 데이터 전처리의 의존도가 매우 적다.
- 학습을 통해 데이터에 있는 규칙을 자동으로 찾아내서 Tree 기반의 분류 규칙을 만든다.
- 각 특성이 개별적으로 처리되어 데이터를 분할하는데 데이터 스케일의 영향을 받지 않으므로 결정 트리에서는 정규화나 표준화 같은 전처리 과정이 필요없다.
- 영향을 가장 많이 미치는 feature를 찾아낼 수도 있다.
- 예측 성능을 계속해서 향상시키면 복잡한 규칙 구조를 가지기 때문에 <sub>※</sub>과적합(Overfitting)이 발생해서 예측 성능이 저하될 수도 있다.
- 가장 상위 노드를 "루트 노드"라고 하며, 나머지 분기점을 "서브 노드", 결정된 분류값 노드를 "리프 노드"라고 한다.

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/17487898-7ca2-4b7d-a465-f98a973ef7fc" width="550px" style="margin: 20px 0 20px 20px">

- 복잡도를 감소시키는 것이 주목적이며, 정보의 복잡도를 불순도(Impurity)라고 한다.
- 이를 수치화한 값으로 지니 계수(Gini coeficient)가 있다.
- 클래스가 섞이지 않고 분류가 잘 되었다면, 불순도 낮다.
- 클래스가 많이 섞여있고 분류가 잘 안되었다면, 불순도 높다.
- 통계적 분산 정도를 정량화하여 표현한 값이고, 0과 1사이의 값을 가진다.
- 지니 계수가 낮을 수록 분류가 잘 된 것이다.
---
<sub>※ 과적합이란, 학습 데이터를 과하게 학습시켜서 실제 데이터에서는 오차가 오히려 증가하는 현상이다.</sub>  

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/a0ec0635-1c0d-41de-8404-f095f1a888c0" width="350px" style="margin-left: -10px">

#### Graphviz
- 결정트리 모델을 시각화할 수 있다.
- https://graphviz.org/download/
  graphviz-9.0.0 (64-bit) EXE installer [sha256]

#### Feature별 중요도
**각 feature가 분류를 나누는 데에 얼마나 큰 영향을 미쳤는지를 표기하는 척도이다.**
- feature_importances_
- 분류를 결정하는 데에 얼만큼 기여했는지를 feature별로 수치를 리턴해준다.

#### 결정 트리의 과적합
- 위에서 알아낸 것처럼 petal length, petal width, 2개의 feature만으로도 분류가 가능하기 때문에, 2차원 산점도를 통해 시각화 할 수 있다.
- 마지막까지 노드가 분리되었기 때문에 과적합이 발생했을 가능성이 있고, 이를 <sub>※</sub>하이퍼 파라미터 튜닝을 통해 해결할 수 있다.

<sub>※ 하이퍼 파라미터란, 최적의 훈련 모델을 구현하기 위해 알고리즘의 수치를 조정할 수 있는 변수를 의미한다.</sub>
---
### 교차 검증 (Cross Validation)
- 기존 방식에서는 데이터 세트에서 학습 데이터 세트와 테스트 데이터 세트를 분리한 뒤 모델 검증을 진행한다.
- 교차 검증 시, 학습 데이터를 다시 분할하여 학습 데이터와 모델 성능을 1차 평가하는 검증 데이터로 나눈다.

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/09b58de5-16b5-4d7a-b680-f9d6c23e3afe" width="500px">

#### 교차 검증의 장단점
- 👍특정 데이터 세트에 대한 과적합 방지
- 👍데이터 세트 규모가 적을 시 과소적합 방지
- 👎모델 훈련, 모델 평가에 소요되는 시간 증가
- 즉, 과적합을 피하고 하이퍼 파라미터를 튜닝함으로써 모델을 일반화하고 신뢰성을 증가시키기 위한 목적이다.

#### 교차 검증의 종류
K-Fold
- k개의 데이터 폴드 세트를 만든 뒤 k번 만큼 학습과 검증 평가를 반복하여 수행하는 방식.
- 학습 데이터와 검증 데이터를 정확히 자르기 때문에 타겟 데이터의 비중이 한 곳으로 치중될 수 있다.
- 예를 들어, 0, 1, 2 중에서 0, 1 두 가지만 잘라서 검증하게 되면 다른 하나의 타겟 데이터를 예측할 수 없게 된다.
- Stratified K-Fold로 해결한다.
  
Stratified K-Fold
- K-Fold와 마찬가지로 k번 수행하지만, 학습 데이터 세트와 검증 데이터 세트가 가지는 타겟 분포도가 유사하도록 검증한다.
- 타겟 데이터의 비중을 항상 똑같게 자르기 때문에 데이터가 한 곳으로 치중되는 것을 방지한다.

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/87e8318c-861c-4b96-a839-73c93eae844f" width="600px">

GridSearchCV
- 교차 검증과 최적의 하이퍼 파라미터 튜닝을 한 번에 할 수 있는 객체이다.
- max_depth와 min_samples_split에 1차원 정수형 list를 전달하면, 2차원으로 결합하여 격자(Grid)를 만들고, 이 중 최적의 점을 찾아낸다.
- 딥러닝에서는 학습 속도가 머신러닝에 비해 느리고, 레이어(층)가 깊어질수록 조정해주어야 할 하이퍼 파라미터 값이 많아지기 때문에, RandomSearchCV에서 대략적인 범위를 찾은 다음, GridSearchCV로 디테일을 조정하는 방식을 사용한다.

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/a0555ad4-01cc-4fef-b780-03e5315217f6" width="500px">

#### 편하게 수행할 수 있는 교차 검증
**cross_val_score(estimator, x, y, cv, scoring)**
- estimator: classifier 종류 모델이면 내부적으로 stratified K-Fold로 진행된다.
- x: features
- y: targets
- cv: 폴드 세트 개수
- scoring: 평가 함수, 정확도(accuracy)외에 다른 것은 다른 장에서 배운다.

#### GridSearchCV
**GridSearchCV(estimator, param_grid, cv, refit, return_train_score)**
- estimator: 학습할 모델 객체 생성
- param_grid: dict형태로 전달해야 하며, 주요 key값은 max_depth, min_samples_split이다.
- cv: 폴드 세트 개수
- refit: 전달한 모델 객체에 최적의 하이퍼 파라미터로 적용하고자 할 때
- return_train_score: 교차 검증 점수를 가져올 지에 대해 True 또는 False를 전달한다.






