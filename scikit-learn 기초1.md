```python
import sys
print("Python 버전: {}".format(sys.version))

import pandas as pd
print("pandas 버전: {}".format(pd.__version__))

import matplotlib
print("matplotlib 버전: {}".format(matplotlib.__version__))

import numpy as np
print("NumPy 버전: {}".format(np.__version__))

import scipy as sp
print("SciPy 버전: {}".format(sp.__version__))

import IPython
print("IPython 버전: {}".format(IPython.__version__))

import sklearn
print("scikit-learn 버전: {}".format(sklearn.__version__))
```

    Python 버전: 3.7.8 | packaged by conda-forge | (default, Jul 31 2020, 01:53:57) [MSC v.1916 64 bit (AMD64)]
    pandas 버전: 1.1.3
    matplotlib 버전: 3.3.2
    NumPy 버전: 1.18.5
    SciPy 버전: 1.5.3
    IPython 버전: 7.18.1
    scikit-learn 버전: 0.23.2
    


```python
# 1.데이터 가져오기
# 아이리스 데이터 로드 및 데이터셋(데이터, 레이블, 특성, 설명) 보기

from sklearn.datasets import load_iris
dataset = load_iris()


print("iris_dataset의 키: \n{}".format(dataset.keys()))    
#data : 값 데이터, target : 레이블 코드(0,1,2), target_names : 레이블 명, DESCR : 설명, feature_names : 특성 명(칼럼)
print(type(dataset))   # sklearn.utils.Bunch

#print(dataset["data"], type(dataset["data"]))   # numpy.ndarray
print(type(dataset["data"]), dataset["data"].shape )  # numpy.ndarray, (150,4)
print("data의 처음 다섯 행:\n{}".format(iris_dataset['data'][:5]))
print(dataset["target"], type(dataset["target"]), dataset["target"].shape )  # numpy.ndarray, (150,)
print(dataset["target_names"], type(dataset["target_names"]), dataset["target_names"].shape)   # numpy.ndarray, (3,)
print(dataset["DESCR"], type(dataset["DESCR"]))   # str
print(dataset["feature_names"], type(dataset["feature_names"]))   
# list, = dataset.feature_names가 동일 pandas의 dataframe과 비슷
print(dataset["filename"], type(dataset["filename"]))   # str

```

    iris_dataset의 키: 
    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
    <class 'sklearn.utils.Bunch'>
    <class 'numpy.ndarray'> (150, 4)
    data의 처음 다섯 행:
    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]]
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2] <class 'numpy.ndarray'> (150,)
    ['setosa' 'versicolor' 'virginica'] <class 'numpy.ndarray'> (3,)
    .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
                    
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
    from Fisher's paper. Note that it's the same as in R, but not as in the UCI
    Machine Learning Repository, which has two wrong data points.
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    .. topic:: References
    
       - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ... <class 'str'>
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] <class 'list'>
    C:\Programs\Anaconda3\envs\tensorflow2.0\lib\site-packages\sklearn\datasets\data\iris.csv <class 'str'>
    


```python
# 2. 학습을 위해 훈련 데이터와 테스트 데이터로 데이터 나누기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
# 데이터는 대문자 X로 표시하고 레이블은 소문자 y로 표기(수학에서 함수의 입력을 x, 출력을 y로 나타내는 표준 공식 f(x)=y에서 유래)
# 데이터는 2차원 배열(행렬)이므로 대문자 X를, 타깃은 1차원 배열(벡터)이므로 소문자 y를 사용
# 여러 번 실행해도 결과가 똑같이 나오도록 유사 난수 생성기에 넣을 난수 초깃값을 random_state 매개변수로 전달,항상 random_state를 고정

print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))
```

    X_train 크기: (112, 4)
    y_train 크기: (112,)
    X_test 크기: (38, 4)
    y_test 크기: (38,)
    


```python
# 3. 데이터 살펴보기 : 데이터프레임으로 변경하여 산점도 행렬로 데이터 분석

import pandas as pd
import seaborn as sns

# X_train 데이터를 사용해서 데이터프레임을 만듭니다.

# 열의 이름은 dataset.feature_names에 있는 문자열을 사용합니다.
dataframe = pd.DataFrame(X_train, columns=dataset.feature_names)

print(dataframe)

# 데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듭니다.
pd.plotting.scatter_matrix(dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8)

dataframe["target"] = pd.Series(dataset.target)
#sns.set(style="ticks", color_codes=True)
g = sns.pairplot(dataframe, hue="target")    #pairplot : 여러 변수간 산점도, hue : 범례

```

         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    0                  5.9               3.0                4.2               1.5
    1                  5.8               2.6                4.0               1.2
    2                  6.8               3.0                5.5               2.1
    3                  4.7               3.2                1.3               0.2
    4                  6.9               3.1                5.1               2.3
    ..                 ...               ...                ...               ...
    107                4.9               3.1                1.5               0.1
    108                6.3               2.9                5.6               1.8
    109                5.8               2.7                4.1               1.0
    110                7.7               3.8                6.7               2.2
    111                4.6               3.2                1.4               0.2
    
    [112 rows x 4 columns]
    


![png](output_3_1.png)



![png](output_3_2.png)



```python
# 4. 첫 번째 머신러닝 모델: k-최근접 이웃 알고리즘 훈련

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# 훈련
knn.fit(X_train, y_train)

```




    KNeighborsClassifier(n_neighbors=1)




```python
# 5. 첫 번째 머신러닝 모델: k-최근접 이웃 알고리즘 예측
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

# 예측
prediction = knn.predict(X_new)
print("예측: {}".format(prediction))
print("예측한 타깃의 이름: {}".format(dataset['target_names'][prediction]))
```

    X_new.shape: (1, 4)
    예측: [0]
    예측한 타깃의 이름: ['setosa']
    


```python
# 6. 모델 평가

# 테스트 데이터 세트 예측
y_pred = knn.predict(X_test)
print("테스트 세트에 대한 예측값:\n {}".format(y_pred))

# 테스트 세트의 정확도
print("테스트 세트의 정확도: {:.2f}".format(np.mean(y_pred == y_test)))
```

    테스트 세트에 대한 예측값:
     [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
     2]
    테스트 세트의 정확도: 0.97
    
