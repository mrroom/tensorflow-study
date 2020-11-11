## 머신러닝 용어
  - 데이터
  - 모델
  - 특징
  - 레이블
  - 파이프라인(과정) : '데이터 넣으면 인텔리전스(지식,정보) 반환', 딥 러닝 모델은 데이터를 넣어주면 결과까지 이어지는 모델을 만들 수 있다. 가장 기본적인 이미지 분류를 생각하면, 이미지의 픽셀값 자체를 큰 전처리 없이 모델에 입력하면 그 이미지의 클래스가 바로 추출되고, 또한 그렇게 되도록 바로 학습, 여러 모델을 조합하는 문제 뿐만 아니라 데이터를 만들고 검수하고 모델을 학습하고 평가하는 과정, 데이터 전처리 => 탐색적 데이터 분석 => 모델 생성 => 모델 훈련 => 모델 검증 => 모델 저장 => 예측 및 서비스(배포, 클라우드/디바이스) => 모니터링, 모델 성능 못지않게 파이프라인 설계가 중요, 파이프라인 관리 툴
  - 파라미터(매개변수) : 모델 내부에서 결정되는 변수 = 데이터로부터 결정되는 변수, 평균(μ)과 표준편차(σ), 선형 회귀의 계수, 사용자에 의해 조정되지 않음, 모델의 능력 결정, 학습된 모델의 일부로 저장
  - 하이퍼 파라미터(초매개변수) : 모델링할 때 사용자가 직접 세팅해주는 값, 학습률(learning rate), 서포트 벡터 머신의 C, sigma값, KNN의 K값, 모델의 파라미터를 조정한다(X) => 모델의 하이퍼 파라미터를 조정한다(O), 정해진 최적의 값이 없고 경험에 의해서, 모델의 정확도를 높이기 위해 학습을 튜닝하는 변수
  - 사전 학습된 이미지 분류(CNN) 모델 ('AlexNet', 'DenseNet', 'GoogLeNet', 'Inception3', 'MobileNetV2', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG', 'alexnet', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'detection', 'googlenet', 'inception', 'inception_v3')  
    ![Pre-trained Models for Image Classification](https://www.learnopencv.com/wp-content/uploads/2019/06/Model_Timeline.png)

## tensorflow 1.x vs 2.x
- ['파이토치' vs. '텐서플로우'··· 딥러닝 프레임워크 선택하기](http://www.ciokorea.com/news/129667)
- [텐서플로우 2.0에서 달라지는 점](https://mc.ai/텐서플로우-2-0에서-달라지는-점/)
- [텐서플로우 1.0 2.0 차이](https://needjarvis.tistory.com/515)
- [텐서플로 1 코드를 텐서플로 2로 바꾸기](https://www.tensorflow.org/guide/migrate?hl=ko)
- [Tensorflow 2.0 변경된 점](https://provia.tistory.com/78)
- [리뷰 | 텐서플로우 2, "더 쉬워진 머신러닝"](http://www.itworld.co.kr/news/125595)
- [[tensoflow 2.0] 2. 텐서플로 사용하기](https://leejigun.github.io/tensorflow2_2)
