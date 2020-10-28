```python
# 행렬의 연산

# 행렬의 곱

import numpy as np


mat_a = np.arange(1,5).reshape(2,2)
mat_b = np.arange(5,9).reshape(2,2)

print(mat_a)
print(mat_b)
print("=================")

mat_c = np.matmul(mat_a, mat_b)
print(mat_c)

# 행렬의 원소간의 곱

print("=================")
mat_c = np.multiply(mat_a, mat_b)
print(mat_c)

mat_c = mat_a * mat_b
print(mat_c)


```

    [[1 2]
     [3 4]]
    [[5 6]
     [7 8]]
    =================
    [[19 22]
     [43 50]]
    =================
    [[ 5 12]
     [21 32]]
    [[ 5 12]
     [21 32]]
    


```python
# 행렬의 연산

# 행렬의 합

import numpy as np

list_a = [[1,3,5],[2,4,6]]
list_b = [[1,2,3],[4,5,6]]

mat_a = np.array(list_a)
mat_b = np.array(list_b)

mat_c = mat_a + mat_b    #axis = 0(row) 기준으로 합산
print(mat_c)

print("=================")

mat_sum = np.arange(9).reshape(3,3)
print(mat_sum)
print()

mat_sum_0 = np.sum(mat_sum, axis=0)   #axis = 0(row) 기준으로 합산
print('axis=0 합')
print(mat_sum_0)
print()

mat_sum_1 = np.sum(mat_sum, axis=1)   #axis = 1(column) 기준으로 합산
print('axis=1 합') 
print(mat_sum_1)

```

    [[ 2  5  8]
     [ 6  9 12]]
    =================
    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    
    axis=0 합
    [ 9 12 15]
    
    axis=1 합
    [ 3 12 21]
    


```python
# 행렬의 연산

# 행렬의 평균

mat_sum = np.arange(9).reshape(3,3)
print(mat_sum)
print()

mat_sum_0 = np.mean(mat_sum, axis=0)    #float으로 변경
print('axis=0 평균')
print(mat_sum_0)
print()

mat_sum_1 = np.mean(mat_sum, axis=1)    #float으로 변경
print('axis=1 평균')
print(mat_sum_1)
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    
    axis=0 평균
    [3. 4. 5.]
    
    axis=1 평균
    [1. 4. 7.]
    
