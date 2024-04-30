<sub>파란색 글씨 클릭 시, 자세한 설명을 작성해놓은 페이지로 이동</sub>
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC">데이터 전처리</a>

#### StandardScaler()
- 데이터의 평균을 0, 분산을 1이 되도록, 표준 정규분포를 따르게 하는 스케일링

#### MinMaxScaler()
- 데이터가 0~1 사이에 위치하도록 최소값은 0, 최대값을 1로 변환한다.

#### MaxAbsScaler()
- 모든 값을 -1~1 사이에 위치하도록, 절대값의 최소값은 0, 최대값은 1이 되도록 변환한다.

#### 로그변환 (Log transformation)
- 왜도와 첨도를 가진 변수를 정규분포에 가깝게 만들어준다. 큰 수치를 같은 비율의 작은 수치로 변환한다.

#### 언더 샘플링 (Under sampling)
- 불균형한 데이터 세트에서 높은 비율을 차지하던 클래스의 데이터 수를 줄임으로써 데이터 불균형을 해소한다.

#### 오버 샘플링 (Over sampling)
- 불균형한 데이터 세트에서 낮은 비율 클래스의 데이터 수를 늘림으로써 데이터 불균형을 해소한다.

#### SMOTE (Synthetic Minority Over-sampling Technique)
- 반드시 학습 데이터 세트만 오버 샘플링 해야 한다.
- 검증 혹은 테스트 데이터 세트를 오버 샘플링하는 경우 원본 데이터가 아닌 데이터에서 검증되기 때문에 올바른 검증이 되지 않는다.
- 낮은 비율 클래스 데이터들의 최근접 이웃을 이용하여 새로운 데이터를 생성한다.

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/AI#ai-artificial-intelligence">AI (Artificial Intelligence)</a>

#### Rule-base AI
- 특정 상황을 이해하는 전문가가 직접 입력값(문제)과 특징을 전달(규칙)하여 출력값(정답)을 내보내는 알고리즘이다.

#### Machine Learning AI
- 데이터를 기반으로 규칙성(패턴)을 학습하여 결과를 추론하는 알고리즘이다.
  
1. 지도 학습 (Supervised Learning)   
> 입력값(문제)과 출력값(정답)을 전달하면, 알아서 특징을 직접 추출하는 방식이다.  
> 다른 입력값(문제)과 동일한 출력값(정답)을 전달하면 새로운 특징을 알아서 추출한다.

2. 비지도 학습 (Unsupervised Learning)   
> 입력값(문제)만 전달하고 정답 없이 특징만 추출하는 학습이다.  
> 추출한 특징에 대해 AI가 출력값(정답)을 부여하여 입력값(문제)은 출력값(정답)이라는 것을 알아낸다.  

3. 강화 학습 (Reinforcement Learning)
   
---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/AI#%EB%B6%84%EB%A5%98-classifier">분류 (Classifier)</a>
- 대표적인 지도학습 방법 중 하나이며, 다양한 문제와 정답을 학습한 뒤 별도의 테스트에서 정답을 예측한다.

#### 피처 (Feature)
- 타겟을 제외한 나머지 속성을 의미한다.
#### 레이블(Label), 클래스(Class), 타겟(Target), 결정(Decision)
- 지도 학습 시, 데이터의 학습을 위해 주어지는 정답을 의미한다.

### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/AI#%EB%B6%84%EB%A5%98-%EC%98%88%EC%B8%A1-%ED%94%84%EB%A1%9C%EC%84%B8%EC%8A%A4">분류 예측 프로세스</a>

#### 데이터 세트 분리
**train_test_split(feature, target, test_size, random_state)**

- 학습 데이터 세트와 테스트 데이터 세트를 분리해준다.

#### 모델 학습
**fit(train_feature, train_target)**
- 모델을 학습시킬 때 사용한다.

#### 평가
**accuracy_score(y_test, predict(X_test))**
- 모델이 얼마나 잘 예측했는지를 "정확도"라는 평가 지표로 평가할 때 사용한다.

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#%EA%B2%B0%EC%A0%95-%ED%8A%B8%EB%A6%AC-decision-tree">결정 트리 (Decision Tree)</a>
- 매우 쉽고 유연하게 적용될 수 있는 알고리즘으로서 데이터의 스케일링, 정규화 등의 데이터 전처리의 의존도가 매우 적다.
- 학습을 통해 데이터에 있는 규칙을 자동으로 찾아내서 Tree 기반의 분류 규칙을 만든다.
- 각 특성이 개별적으로 처리되어 데이터를 분할하는데 데이터 스케일의 영향을 받지 않으므로 결정 트리에서는 정규화나 표준화 같은 전처리 과정이 필요없다.
- 영향을 가장 많이 미치는 feature를 찾아낼 수도 있다.
- 예측 성능을 계속해서 향상시키면 복잡한 규칙 구조를 가지기 때문에 <sub>※</sub>과적합(Overfitting)이 발생해서 예측 성능이 저하될 수도 있다.
- 가장 상위 노드를 "루트 노드"라고 하며, 나머지 분기점을 "서브 노드", 결정된 분류값 노드를 "리프 노드"라고 한다.

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/17487898-7ca2-4b7d-a465-f98a973ef7fc" width="550px" style="margin: 20px 0 20px 20px">

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#%EA%B5%90%EC%B0%A8-%EA%B2%80%EC%A6%9D-cross-validation">교차 검증 (Cross Validation)</a>
- 기존 방식에서는 데이터 세트에서 학습 데이터 세트와 테스트 데이터 세트를 분리한 뒤 모델 검증을 진행한다.
- 교차 검증 시, 학습 데이터를 다시 분할하여 학습 데이터와 모델 성능을 1차 평가하는 검증 데이터로 나눈다.

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/09b58de5-16b5-4d7a-b680-f9d6c23e3afe" width="500px">

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#%EB%B6%84%EB%A5%98-classification-%EC%84%B1%EB%8A%A5-%ED%8F%89%EA%B0%80-%EC%A7%80%ED%91%9C">분류 (Classification) 성능 평가 지표</a>

#### 정확도 (Accuracy)
#### 오차 행렬 (Confusion Matrix)
#### 정밀도 (Precision)
#### 재현율 (Recall)
#### 정밀도와 재현율의 트레이드 오프 (Trade-off)
- 분류 시, 결정 임계값(Threshold)을 조정해서 정밀도 또는 재현율의 수치를 높일 수 있다.
#### F1 Score
- F1 Score는 0~1까지 점수를 매길 수 있으며, 0에 가까울 수록 정밀도와 재현율 모두 낮다는 뜻이다.
#### ROC Curve, AUC

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#%EB%B2%A0%EC%9D%B4%EC%A6%88-%EC%B6%94%EB%A1%A0-%EB%B2%A0%EC%9D%B4%EC%A6%88-%EC%A0%95%EB%A6%AC-%EB%B2%A0%EC%9D%B4%EC%A6%88-%EC%B6%94%EC%A0%95-bayesian-inference">베이즈 추론, 베이즈 정리, 베이즈 추정 (Bayesian Inference)</a>
- 역확률(inverse probability) 문제를 해결하기 위한 방법으로서, 조건부 확률(P(B|A)))을 알고 있을 때, 정반대인 조건부 확률(P(A|B))을 구하는 방법이다.
- 추론 대상의 사전 확률과 추가적인 정보를 기반으로 해당 대상의 "사후 확률"을 추론하는 통계적 방법이다.
- 어떤 사건이 서로 "배반"하는(독립하는) 원인 둘에 의해 일어난다고 하면, 실제 사건이 일어났을 때 이 사건이 두 원인 중 하나일 확률을 구하는 방식이다.
- 어떤 상황에서 N개의 원인이 있을 때, 실제 사건이 발생하면 N개 중 한 가지 원인일 확률을 구하는 방법이다.
- 기존 사건들의 확률을 알 수 없을 때, 전혀 사용할 수 없는 방식이다.
- 하지만, 그 간 데이터가 쌓이면서, 기존 사건들의 확률을 대략적으로 뽑아낼 수 있게 되었다.
- 이로 인해, 사회적 통계나 주식에서 베이즈 정리 활용이 필수로 꼽히고 있다.  

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#%EB%82%98%EC%9D%B4%EB%B8%8C-%EB%B2%A0%EC%9D%B4%EC%A6%88-%EB%B6%84%EB%A5%98-naive-bayes-classifier">나이브 베이즈 분류 (Naive Bayes Classifier)</a>
- 텍스트 분류를 위해 전통적으로 사용되는 분류기로서, 분류에 있어서 준수한 성능을 보인다.
- 베이즈 정리에 기반한 통계적 분류 기법으로서, 정확성도 높고 대용량 데이터에 대한 속도도 빠르다.
- 반드시 모든 feature가 서로 독립적이여야 한다. 즉, 서로 영향을 미치지 않는 feature들로 구성되어야 한다.
- 감정 분석, 스팸 메일 필터링, 텍스트 분류, 추천 시스템 등 여러 서비스에서 활용되는 분류 기법이다.
- 빠르고 정확하고 간단한 분류 방법이지만, 실제 데이터에서  
  모든 feature가 독립적인 경우는 드물기 때문에 실생활에 적용하기 어려운 점이 있다.

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0-svm-support-vector-machine">서포트 벡터 머신 (SVM, Support Vector Machine)</a>
- 기존의 분류 방법들은 '오류율 최소화'의 목적으로 설계되었다면, SVM은 두 부류 사이에 존재하는 '여백 최대화'의 목적으로 설계되었다.
- 분류 문제를 해결하는 지도 학습 모델 중 하나이며, 결정 경계라는 데이터 간 경계를 정의함으로써 분류를 할 수 있다.
- 새로운 데이터가 경계를 기준으로 어떤 방향에 잡히는지를 확인함으로써 해당 데이터의 카테고리를 예측할 수 있다.
- 데이터가 어느 카테고리에 속할지 판단하기 위해 가장 적절한 경계인 결정 경계를 찾는 선형 모델이다.  

<img src="https://github.com/SOYOUNGdev/study-machine_learning/assets/115638411/853a6d7e-5e9c-4278-a881-86076850dcfa" width="400px" style="margin-bottom: 60px;">

#### 서포트 벡터 (Support Vector)
#### 결정 경계 (Decision boundary)
#### 하드 마진(Hard margin)
- 매우 엄격하게 집단을 구분하는 방법으로 이상치를 허용해주지 않는 방법이다.
#### 소프트 마진(Soft margin)
- 이상치를 허용해서 일부 데이터를 잘못 분류하더라도 나머지 데이터를 더욱 잘 분류해주는 방법이다.
#### 커널 트릭 (Kernel trick)
- 선형으로 완전히 분류할 수 없는 데이터 분포가 있을 경우 소프트 마진을 통해 어느정도 오류는 허용하는 형태로 분류할 수는 있다.  
하지만, 더 잘 분류하기 위해서는 차원을 높여야 한다. 이를 고차원 매핑이라고 하고 이 때 커널 트릭을 사용한다.
- 비선형 데이터일 때 RBF 커널을 사용하고, 선형 데이터일 때 linear 커널을 사용하는 것이 효과적이다.

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#feature-selection">Feature Selection</a>
- 결과 예측에 있어서, 불필요한 feature들로 인해 모델 예측 성능을 떨어뜨릴 가능성을 사전 제거할 수 있다.
- 타겟 데이터와 관련이 없는 feature들을 제거하여, 타겟 데이터를 가장 잘 예측하는 feature들의 조합(상관관계가 높은)을 찾아내는 것이 목적이다.

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#k-%EC%B5%9C%EA%B7%BC%EC%A0%91-%EC%9D%B4%EC%9B%83-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-k-nearest-neighbor-knn">K-최근접 이웃 알고리즘 (K-Nearest Neighbor, KNN)</a>
- 임의의 데이터가 주어지면 그 주변(이웃)의 데이터를 살펴본 뒤 더 많은 데이터가 포함되어 있는 범주로 분류하는 방식이다.
- 가장 간단한 머신러닝 알고리즘으로서, 직관적이고 나름 성능도 괜찮다.
- K를 어떻게 정하는지에 따라서 결과값이 바뀔 수 있다. K는 임의의 데이터가 주어졌을 때 가까운 이웃들의 개수이고 기본값은 5이다.
- K는 가장 가까운 5개의 이웃 데이터를 기반으로 분류하며, 일반적으로 홀수를 사용한다. 짝수일 경우 동점이 되어 하나의 결과를 도출할 수 없기 때문이다.

---
### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5-ensemble-learning">앙상블 학습 (Ensemble Learning)</a>
- 어떤 데이터의 값을 예측한다고 할 때, 하나의 모델만 가지고 결과를 도출할 수도 있지만, 여러개의 모델을 조화롭게 학습시켜 그 모델들의 예측 결과들을 이용한다면, 더 정확한 예측값을 구할 수 있다.
- 여러 개의 분류기를 생성하고 그 예측을 결합하여 1개의 분류기를 사용할 때 보다 더 정확하고 신뢰성 높은 예측을 도출하는 기법이다.
- 강력한 하나의 모델을 사용하는 것보다 약한 모델을 여러 개 조합하여 더 정확한 예측에 도움을 주는 방식이다.
- 앙상블 학습의 주요 방법은 배깅(Baggin)과 부스팅(Boosting)이다.

#### 보팅(Voting)
- "하나의 데이터 세트"에 대해 서로 다른 알고리즘을 가진 분류기를 결합하는 방식이다.
- 서로 다른 분류기들에 "동일한 데이터 세트"를 병렬로 학습해서 예측값을 도출하고, 이를 합산하여 최종 예측값을 산출해내는 방식을 말한다.

> 1. 하드 보팅 (Hard Voting)
> - 각 분류기가 만든 예측값을 다수결로 투표해서 가장 많은 표를 얻은 예측값을 최종 예측값으로 결정하는 보팅 방식을 말한다.

> 2. 소프트 보팅 (Soft Voting)
> - 각 분류기가 예측한 타겟별 확률을 평균내어 가장 높은 확률의 타겟을 최종 예측값으로 도출한다.

#### 배깅 (Bagging, Bootstrap Aggregation)
- 하나의 데이터 세트에서 "여러 번 중복을 허용하면서 학습 데이터 세트를 랜덤하게 뽑은 뒤(Bootstrap)" 하나의 예측기 여러 개를 병렬로 학습시켜서 결과물을 집계(Aggregation)하는 방법이다.
- Voting 방식과 달리 같은 알고리즘의 분류기를 사용하고 훈련 세트를 무작위로 구성하여 각기 다르게(독립적으로, 병렬로) 학습시킨다.
- 학습 데이터가 충분하지 않더라도 충분한 학습효과를 주어 과적합등의 문제를 해결하는 데 도움을 준다.
- 배깅 방식을 사용한 대표적인 알고리즘이 바로 랜덤 포레스트 알고리즘이다.

#### 부스팅(Boosting)
- 이전 분류기의 학습 결과를 토대로 다음 분류기의 학습 데이터의 샘플 가중치를 조정해서 "순차적으로" 학습을 진행하는 방법이다.
- 이전 분류기를 계속 개선해 나가는 방향으로 학습이 진행되고, 오답에 대한 높은 가중치를 부여하므로 정확도가 높게 나타난다.
- 높은 가중치를 부여하기 때문에 이상치(Outlier)에 취약할 수 있다.

> 1. Adaboost(Adaptive boosting)
> - 부스팅에서 가장 기본 기법이며, 결정 트리와 비슷한 알고리즘을 사용하지만 뻗어나가지(tree) 않고 하나의 조건식만 사용(stump)하여 결정한다.
> - 여러 개의 stump로 구성되어 있으며, 이를 Forest of stumps라고 한다.

> 2. GBM(Gradient Boost Machine)
> - Adaboost와 유사하지만, 에러를 최소화하기 위해 가중치를 업데이트할 때 경사 하강법(Gradient Descent)을 이용한다.
> - GBM은 과적합에도 강하고 뛰어난 성능을 보이지만, 병렬 처리가 되지 않아서 수행 시간이 오래 걸린다는 단점이 있다.
> - 경사 하강법이란, 오류를 최소화하기 위해 Loss function의 최소값까지 점차 하강하면서 찾아나가는 기법이다.
> - 모델 A를 통해 y를 예측하고 남은 잔차(residual, 에러의 비율)를 다시 B라는 모델을 통해 예측하고 A + B모델을 통해 y를 예측하는 방식이다.

> 3. XGBoost(eXtra Gradient Boost)
> 트리 기반의 앙상블 학습에서 가장 각광받고 있는 알고리즘 중 하나이며, 분류에 있어서 일반적으로 다른 머신러닝보다 뛰어난 예측 성능을 나타낸다.
> - GBM에 기반하고 있지만 병렬 CPU 환경에서 병렬 학습이 가능하기 때문에 기존 GBM보다 빠르게 학습을 완료할 수 있다.
> - 하이퍼 파라미터를 조정하여 분할 깊이를 변경할 수 있지만, tree pruning(가지치기)으로 더 이상 긍정 이득이 없는 분할을 가지치기해서 분할 수를 줄이는 추가적인 장점을 가지고 있다.

> 4. LightGBM(Light Gradient Boosting Machine)
> - XGBoost의 향상된 버전으로서 결정트리 알고리즘을 기반으로 순위 지정, 분류 및 기타 여러 기계 학습 작업에 사용할 수 있다.
> - 기존 부스팅 방식과 마찬가지로 각각의 새로운 분류기각 이전 트리의 잔차를 조정해서 모델이 향상되는 방식으로 결합되고, 마지막으로 추가된 트리는 각 단계의 결과를 집계하여 강력한 분류기가 될 수 있다.
> - XGBoost와 달리 GOSS 알고리즘을 사용해서 수직으로 트리를 성장시킨다. 즉, 다른 알고리즘은 레벨(depth)단위로 성장시키지만, LightGBM은 리프(leaf) 단위로 성장시킨다.
> - 인코딩을 따로 할 필요 없이 카테고리형 feature를 최적으로 변환하고 이에 따른 노드 분할을 수행한다.  
>   astype('category')로 변환할 수 있으며, 이는 다른 다양한 인코딩 방식보다 월등히 우수하다.  

---
#### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Model#%EB%B3%B4%ED%8C%85-voting">보팅 (Voting)</a>
**VotingClassifier(n_estimators, voting)**
- n_estimators: 추가할 모델 객체를 list형태로 전달한다. 각 모델은 튜플 형태인 ('key', model)로 작성한다.
- voting: 'soft', 'hard' 둘 중 선택한다(default: 'hard')

#### 배깅(Bagging) - 랜덤 포레스트(Random Forest)
**RandomForestClassifier(n_estimators, min_samples_split, min_samples_leaf, n_jobs)**
- n_estimators: 생성할 tree(모델)의 개수를 작성한다(default: 50)

#### 부스팅(Boosting)
**AdaBoostClassifier(base_estimators, n_estimators, learning_rate)**
- base_estimators: 학습에 사용하는 알고리즘을 선택한다(default: DecisionTreeClassifier(max_depth=1)).
- n_estimators: 생성할 약한 학습기의 개수를 지정한다(default: 50).
- learning_rate: 학습을 진행할 때마다 적용하는 학습률(0~1사이의 값), 약한 학습기가 순차적으로 오류값을 보정해나갈 때 적용하는 계수이며, 낮은 만큼 최소 손실값을 찾아 예측성능이 높아질 수 있지만, 그 만큼 많은 수의 트리가 필요하고 시간이 많이 소요된다(default: 1)

#### 부스팅(Boosting) - GBM(Gradient Boosting Machine)
**GradientBoostingClassifier(n_estimators, loss, learning_rate,, subsample)**
- n_estimators: 약한 학습기의 개수이며, 많을수록 일정 수준까지는 좋아지지만 그만큼 시간도 오래걸리고 과적합의 위험이 있다.
- loss: 경사 하강법에서 사용할 loss function을 지정한다(default: 'log_loss'). 만약 지수적 감쇠를 사용하고자 한다면, 'exponential'을 지정한다.
- learning_rate: 학습을 진행할 때마다 적용하는 학습률(0~1사이의 값), 약한 학습기가 순차적으로 오류값을 보정해나갈 때 적용하는 계수이며, 낮은 만큼 최소 손실값을 찾아 예측성능이 높아질 수 있지만, 그 만큼 많은 수의 트리가 필요하고 시간이 많이 소요된다(default: 1)
- subsample: 학습에 사용하는 데이터의 샘플링 비율이다.(default: 1(100%)). 과적합 방지 시 1보다 작은 값으로 설정한다.

#### 부스팅(Boosting) - XGBoost(eXtra Gradient Boost)
**XGBClassifier(n_estimators, learning_rate, subsample, eval_set, early_stopping_rounds)**
- eval_set: 예측 오류값을 줄일 수 있도록 반복하면서 학습이 진행되는데, 이 때 학습은 학습 데이터로 하고 예측 오류값 평가는 eval_set으로 지정된 검증 세트로 평가한다.
- early_stopping_rounds: 지정한 횟수동안 오류가 개선되지 않으면 더 이상 학습은 진행하지 않는다.

#### 부스팅(Boosting) - LightGBM(Light Gradient Boosting Machine)
**LGBMClassifier(n_estimators, learning_rate, subsmaple, eval_set)**
- n_estimators: default: 100

---
<details>
  <summary><a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Regression-(%ED%9A%8C%EA%B7%80)">회귀(Regressionn)</a></summary>
      - 데이터가 평균과 같은 일정한 값으로 돌아가려는 경향을 이용한 통계학 기법이다.<br>  
      - 여러 개의 독립 변수와 한 개의 종속 변수 간의 상관관계를 모델링하는 기법을 통칭한다.<br>    
      - feature와 target 데이터 기반으로 학습하여 최적의 회귀 계수(W)를 찾는 것이 회귀의 목적이다.<br>     

  ### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Regression-(%ED%9A%8C%EA%B7%80)#mini-batch">Mini batch</a>
- 전체 데이터를 대상으로 한 번에 경사 하강법을 수행하는 방법은 '배치 경사 하강법'이라 한다.
- 일반적인 배치 경사 하강법은 시간이 너무 오래 걸리기 때문에, 나누어서 하는 방법이 필요하고 이를 '미니 배치 경사 하강법'이라 한다.
- 미니 배치 경사 하강법은 미니 배치 단위로 경사 하강법을 수행하는 방법이다.
</details>

### <a href="https://github.com/SOYOUNGdev/study-machine_learning/wiki/Regression-(%ED%9A%8C%EA%B7%80)#decision-tree-regression-%ED%9A%8C%EA%B7%80-%ED%8A%B8%EB%A6%AC">Decision Tree Regression (회귀 트리)</a>
- 결정 트리와 결정 트리 기반의 앙상블 알고리즘은 분류뿐 아니라 회귀분석도 가능하다.
- 분류와 유사하게 분할하며, 최종 분할 후 각 분할 영역에서 실제 데이터까지의 거리들의 평균 값으로 학습 및 예측을 수행한다.
