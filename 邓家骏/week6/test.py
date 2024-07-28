import numpy as np

# 生成一个 2x3x4 的自然数组
array = np.arange(12).reshape(3,4)
arr_ones = np.ones(3).reshape(3,1)


print(array,arr_ones,array-arr_ones)

