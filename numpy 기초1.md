```python
# Numpy 기초
# 행렬 생성(row,column)
# 행렬 생성 방법 : List로 생성, 특정값으로 생성, 무작위 값으로 생성

# List로 생성

import numpy as np

sample_list = [[1,2,3],[4,5,6],[7,8,9]]
mat_int = np.array(sample_list, dtype=int)
mat_float = np.array(sample_list, dtype=float)

print(mat_int)
print(mat_float)

```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    [[1. 2. 3.]
     [4. 5. 6.]
     [7. 8. 9.]]
    


```python
# Numpy 기초
# 행렬 생성(row,column)
# 행렬 생성 방법 : List로 생성, 특정값으로 생성, 무작위 값으로 생성

# 특정값으로 생성(0,1,특정동일값, 대각행렬값(one hot))

import numpy as np

mat_zero = np.zeros([3,4])
mat_one = np.ones([2,7])
mat_full = np.full([7,3], 10)
mat_eye = np.eye(4)

print(mat_zero)
print(mat_one)
print(mat_full)
print(mat_eye)
```

    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    [[1. 1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1. 1.]]
    [[10 10 10]
     [10 10 10]
     [10 10 10]
     [10 10 10]
     [10 10 10]
     [10 10 10]
     [10 10 10]]
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    


```python
# Numpy 기초
# 행렬 생성(row,column)
# 행렬 생성 방법 : List로 생성, 특정값으로 생성, 무작위 값으로 생성

# 무작위값으로 생성(정규분포방법, 범위지정방법)

import numpy as np

np.random.seed(123)

#mat_rand_normal = np.random.normal(0,1,[2,3])  #평균:0, 표준편차:1 인 2x3 행렬
mat_rand_normal = np.random.normal(85,5,[2,3])  #평균:85, 표준편차:5 인 2x3 행렬
mat_rand_uniform = np.random.uniform(1,20,[2,3])  #min:1, max:20 인 2x3 행렬

print(mat_rand_normal)
print(mat_rand_uniform)

```

    [[79.57184698 89.98672723 86.41489249]
     [77.46852643 82.10699874 93.25718269]]
    [[19.63451977 14.01176503 10.13770613]
     [ 8.45023285  7.52038231 14.85194444]]
    


```python
# Numpy 기초
# 행렬 조작(모양 : Shape, 변환(전치행렬) : Transpose, 범위추출 : Indexing, 차원확장 : Expand Dimension, 차원축소 : Squeeze)

import numpy as np

# 모양

sample_list = [[1,2,3],[4,5,6]]
mat_int = np.array(sample_list, dtype=int)

print(mat_int.shape)

# 전치행렬
mat_one1 = np.ones([2,7])
mat_one1_t = np.transpose(mat_one1)
mat_one2 = np.ones([3,4,5])
mat_one2_t = np.transpose(mat_one2)
mat_one3_t = np.transpose(mat_one2, axes=[2,0,1])

print(mat_one1.shape)
print(mat_one1_t.shape)
print(mat_one2.shape)
print(mat_one2_t.shape)
print(mat_one3_t.shape)

# 범위추출
sample_list = [[1,2,3],[4,5,6]]
mat_int = np.array(sample_list)
print("원소 추출 : ", mat_int[1,1]) # row : 1, column : 1
print("1행 추출 : ", mat_int[0,:]) # row : 0, column : all
print("3열 추출 : ", mat_int[:,2]) # row : all, column : 2
print("행열 추출 : ", mat_int[0:2,1:3]) # 항상 파이썬에서 구간을 지정할때는 시작은 0부터 이며 끝은 미포함(index+1)

# 차원확장
mat_322 = np.ones([3,2,2])    #3차원 : 3 x 2 x 2
mat_1322 = np.expand_dims(mat_322, axis=0)   # 사차원 : 1 x 3 x 2 x 2
mat_3221 = np.expand_dims(mat_322, axis=-1)   # 사차원 : 3 x 2 x 2 x 1
mat_13212 = np.expand_dims(mat_1322, axis=3)   # 오차원 : 1 x 3 x 2 x 1 x 2

print(mat_322.shape)
print(mat_1322.shape)
print(mat_3221.shape)
print(mat_13212.shape)


# 차원축소 :  axis를 안주면, 전체 차원 중 1인 애들을 다 축소 시켜주고, 값을 주면 해당 위치만 축소시켜 줍니다.
mat_1312 = np.ones([1,3,1,2])    #3차원 : 3 x 2 x 2
mat_squ_all = np.squeeze(mat_1312)
mat_squ_1 = np.squeeze(mat_1312, axis=0)
mat_squ_2 = np.squeeze(mat_1312, axis=2)
print(mat_squ_all.shape)    # 2차원 : 3 x 2
print(mat_squ_1.shape)    # 3차원 : 3 x 1 x 2
print(mat_squ_2.shape)    # 3차원 : 1 x 3 x 2


```

    (2, 3)
    (2, 7)
    (7, 2)
    (3, 4, 5)
    (5, 4, 3)
    (5, 3, 4)
    원소 추출 :  5
    1행 추출 :  [1 2 3]
    3열 추출 :  [3 6]
    행열 추출 :  [[2 3]
     [5 6]]
    (3, 2, 2)
    (1, 3, 2, 2)
    (3, 2, 2, 1)
    (1, 3, 2, 1, 2)
    (3, 2)
    (3, 1, 2)
    (1, 3, 2)
    
