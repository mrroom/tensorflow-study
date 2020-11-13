## 머신러닝 용어
  - 데이터
  - 모델
  - 특징
  - 레이블
  - 파이프라인(과정) : '데이터 넣으면 인텔리전스(지식,정보) 반환', 딥 러닝 모델은 데이터를 넣어주면 결과까지 이어지는 모델을 만들 수 있다. 가장 기본적인 이미지 분류를 생각하면, 이미지의 픽셀값 자체를 큰 전처리 없이 모델에 입력하면 그 이미지의 클래스가 바로 추출되고, 또한 그렇게 되도록 바로 학습, 여러 모델을 조합하는 문제 뿐만 아니라 데이터를 만들고 검수하고 모델을 학습하고 평가하는 과정, 데이터 전처리 => 탐색적 데이터 분석 => 모델 생성 => 모델 훈련 => 모델 검증 => 모델 저장 => 예측 및 서비스(배포, 클라우드/디바이스) => 모니터링, 모델 성능 못지않게 파이프라인 설계가 중요, 파이프라인 관리 툴
  - 파라미터(매개변수) : 모델 내부에서 결정되는 변수 = 데이터로부터 결정되는 변수, 평균(μ)과 표준편차(σ), 선형 회귀의 가중치 또는 계수, SVM의 서포트 벡터, 사용자에 의해 조정되지 않음, 모델의 능력 결정, 학습된 모델의 일부로 저장
  - 하이퍼 파라미터(초매개변수) : 모델링할 때 사용자가 직접 세팅해주는 값, 학습률(learning rate), 서포트 벡터 머신의 C, sigma값, KNN의 K값, 모델의 파라미터를 조정한다(X) => 모델의 하이퍼 파라미터를 조정한다(O), 정해진 최적의 값이 없고 경험에 의해서, 모델의 정확도를 높이기 위해 학습을 튜닝하는 변수
  - 사전 학습된 이미지 분류(CNN) 모델 : 'AlexNet', 'DenseNet', 'GoogLeNet', 'Inception3', 'MobileNetV2', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG', 'alexnet', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'detection', 'googlenet', 'inception', 'inception_v3' 
  
    ![Pre-trained Models for Image Classification](https://www.learnopencv.com/wp-content/uploads/2019/06/Model_Timeline.png)
  - 전이학습(Transfer Learning) : 사과 깎는 방법을 익힌 AI에게 배를 깎도록 하는 것, 비가 올 확률을 예측하는 AI에게 눈이 올 확률을 예측, 구글의 티처블 머신, Toonify Yourself
    - 특징 추출을 위해 사전 훈련된 모델 사용 : 사전 훈련된 모델을 인스턴스화하고 완전히 연결된 분류기를 맨 위에 추가. 사전 훈련된 모델은 고정되고 분류기의 가중치만 훈련 중에 업데이트. 컨벌루션 베이스 모델은 각 이미지와 관련된 모든 특징을 추출하며 주어진 추출된 특징을 가지고 이미지 클래스를 결정하는 분류기를 훈련
    - 사전 훈련된 모델을 미세 조정 : 성능을 더욱 향상시키기 위해 사전 훈련된 모델의 최상위 계층을 미세 조정을 통해 새 데이터셋으로써 재사용. 모델이 주어진 데이터셋에 맞는 상위 레벨의 특징을 학습 할 수 있도록 가중치를 조정 일반적으로 훈련 데이터셋의 규모가 크고, 사전 훈련된 모델이 사용했던 원래 데이터셋과 매우 유사한 경우에 권장
  - 모델 개선
    - 데이터 추가 : 모델이 더 많은 예제를 학습할수록 성능이 향상
    - 데이터 보강(증강) : 데이터에 다양성을 추가하는 일반적인 방법은 이미지에 다양한 변환(회전, 대칭 이동, 이동, 자르기)을 적용하여 데이터를 보강
    - 학습 시간 연장 : 학습 시간이 길수록 모델이 더 튜닝됩니다. Epoch 수를 늘리면 모델의 성능이 향상
    - 하이퍼 매개 변수로 실험 : 사용된 매개 변수 외에 다른 매개 변수를 튜닝하여 성능을 개선. 각 Epoch 후의 모델 업데이트 크기를 결정하는 학습 속도를 변경하면 성능이 향상
    - 다른 모델 아키텍처 사용 : 데이터 모양에 따라 기능 학습에 가장 적합한 모델이 다를 수 있어 모델의 성능이 만족스럽지 않으면 아키텍처를 변경
  - 자연어처리(NLP) : 자연어이해(NLU), 자연어생성(NLG), 사람 <-> 기계
  
    ![NLP](http://t1.kakaocdn.net/braincloud/homepage/article_image/244ee65c-6b3b-45cf-8b95-b684343a745c.png)
    
  - 사전 학습된 딥러닝 기반 언어 모델 : ELMo, GPT-1, BERT, RoBERTa, ALBERT, T5
  - 레이어
    -  Dense 레이어(전결합층) : 입출력을 모두 연결, 예를 들어 입력 뉴런이 4개, 출력 뉴런이 8개있다면 총 연결선은 32개(4x8=32) 입니다. 각 연결선에는 가중치(weight)를 포함하고 있는데, 이 가중치가 나타내는 의미는 연결강도라고 보시면 됩니다. 현재 연결선이 32개이므로 가중치도 32개입니다.  
      ex) Dense(8, input_dim=4, init='uniform', activation='relu'))   
      # 1) 출력 뉴런의 수, 2) 입력 뉴런의 수, 3) 가중치 초기화 방법(uniform(균일), normal(정규)), 4) 활성화 함수(linear(기본), relu(은닉층), sigmoid(이진분류), softmax(다중클래스분류)))  
      ex) Dense(1, input_dim=3, activation='sigmoid')) # 출력(이진분류),   
        Dense(3, input_dim=4, activation='softmax')) # 출력(다중클래스분류),   
        Dense(4, input_dim=6, activation='relu')) # 은닉  
    - [컨볼루션 신경망 레이어](https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/) 

## tensorflow 1.x vs 2.x
- ['파이토치' vs. '텐서플로우'··· 딥러닝 프레임워크 선택하기](http://www.ciokorea.com/news/129667)
- [텐서플로우 2.0에서 달라지는 점](https://mc.ai/텐서플로우-2-0에서-달라지는-점/)
- [텐서플로우 1.0 2.0 차이](https://needjarvis.tistory.com/515)
- [텐서플로 1 코드를 텐서플로 2로 바꾸기](https://www.tensorflow.org/guide/migrate?hl=ko)
- [Tensorflow 2.0 변경된 점](https://provia.tistory.com/78)
- [리뷰 | 텐서플로우 2, "더 쉬워진 머신러닝"](http://www.itworld.co.kr/news/125595)
- [[tensoflow 2.0] 2. 텐서플로 사용하기](https://leejigun.github.io/tensorflow2_2)
