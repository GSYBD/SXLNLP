import numpy as np

"""
三、矩阵（Matrix）
    矩阵是一个二维数组
    矩阵就是由多个向量组成的
"""


def generate_matrix():
    print("一维数组", np.array([1, 2, 3]))
    print("二维数组", np.array([[1, 2, 3], [4, 5, 6]]))
    print(np.random.rand(2, 3), "随机生成一个二维数组(2*3)")
    print(np.random.randint(1, 10, (3, 4)), "随机生成一个二维数组(3*4),值[1,10)")
    print(np.random.rand(3, 4).shape, "获取矩阵的维度")


def matrix_operation():
    """
    矩阵的运算
    """
    dimension_m = 3
    dimension_n = 4
    dimension_p = 5
    dimension_q = 3
    matrix0 = np.array([[1], [2], [3]])
    matrix1 = np.random.randint(1, 10, (dimension_m, dimension_n))
    print(matrix1, "矩阵1：", matrix1.shape, type(matrix1))
    matrix2 = np.random.randint(1, 10, (dimension_m, dimension_n))
    print(matrix2, "矩阵2：", matrix2.shape, type(matrix2))
    matrix3 = np.random.randint(1, 10, (dimension_n, dimension_p))
    print(matrix3, "矩阵3：", matrix3.shape, type(matrix3))
    matrix4 = np.random.randint(1, 10, (dimension_p, dimension_q))
    print(matrix4, "矩阵4：", matrix4.shape, type(matrix4))
    print(matrix1 + matrix2, "\n矩阵加法(需要维度相同)，对位相加")
    matrix_dot = np.dot(matrix1, matrix3)
    print(matrix_dot, "\n矩阵乘法 M*N的矩阵乘以N*P的矩阵，返回M*P的矩阵", matrix_dot.shape, type(matrix_dot))
    print("**" * 10, "矩阵的遍历", "***" * 20)
    shape_m = matrix1.shape[0]
    shape_n = matrix1.shape[1]
    shape_p = matrix3.shape[1]
    matrix = np.empty((shape_m, shape_p))
    for m in range(shape_m):
        row = np.empty(shape_p)
        for p in range(shape_p):
            data = int()
            for n in range(shape_n):
                data += matrix1[m][n] * matrix3[n][p]
            row[p] = data
        matrix[m] = row
    print(matrix, "\n验证矩阵乘法(公式计算)", matrix.shape, type(matrix))
    print("**" * 10, "矩阵数学规律 矩阵满足交换律", "***" * 20)
    matrix_exchange1 = np.dot((matrix1 + matrix2), matrix3)
    print(matrix_exchange1, "\n交换前", matrix_exchange1.shape, type(matrix_exchange1))
    matrix_exchange2 = np.dot(matrix1, matrix3) + np.dot(matrix2, matrix3)
    print(matrix_exchange2, "\n交换后", matrix_exchange2.shape, type(matrix_exchange2))
    print("**" * 10, "矩阵数学规律 矩阵满足结合律", "***" * 20)
    matrix_combine1 = np.dot(np.dot(matrix1, matrix3), matrix4)
    print(matrix_combine1, "\n结合前", matrix_combine1.shape, type(matrix_combine1))
    matrix_combine2 = np.dot(matrix1, np.dot(matrix3, matrix4))
    print(matrix_combine2, "\n结合后", matrix_combine2.shape, type(matrix_combine2))
    print("**" * 10, "矩阵的点乘（需要维度相同）", "***" * 20)
    print(matrix0)
    print(matrix1)
    print(np.multiply(matrix0, matrix1), "\n矩阵点乘(对位相乘)")
    print(matrix1)
    print(matrix2)
    print(np.multiply(matrix1, matrix2), "\n矩阵点乘(对位相乘)")
    print("**" * 10, "矩阵转置(transpose或T)", "***" * 20)
    print(matrix1, "\n转置前")
    print(matrix1.transpose(), "\n转置后")
    print(matrix1.T, "\n转置后")
    print("**" * 10, "矩阵与向量的互相转换", "***" * 20)
    print(matrix1, "\n矩阵1", matrix1.shape, type(matrix1))
    flatten = matrix1.flatten()
    print(flatten,"\n矩阵转向量（flatten）")
    print(flatten.reshape(2, 6), "\n向量转矩阵（reshape）")
    print(matrix1.reshape(2, 6), "\n矩阵转矩阵（reshape）")


if __name__ == '__main__':
    '矩阵定义'
    # generate_matrix()
    '矩阵运算'
    matrix_operation()