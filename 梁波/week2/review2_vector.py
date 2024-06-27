import numpy as np

'''
二、向量（Vector）
    一个向量就是一列数
    [1,2,3,4] 四维向量
'''


def generate_vector():
    # 指定向量内容
    vector1 = np.array([1, 2, 3, 4])
    print(vector1, type(vector1))

    # 随机生成一个四维向量,值在[0,1)之间
    vector2 = np.random.rand(4)
    print(vector2, type(vector2))

    # 随机生成一个四维向量，值在[1,10]之间
    vector3 = np.random.uniform(1, 10, 4)
    print(vector3, type(vector3))

    # 随机生成一个四维向量，值在[1,10)之间整数
    vector4 = np.random.randint(1, 10, 4)
    print(vector4, type(vector4))


'''
向量的运算
    1.1 向量加和
    1.2 向量内积
    1.3 向量的模
    1.4 向量夹角余弦值
'''


def vector_operation():
    """
    向量的运算
    """
    dimension = 4  # 维度需要相同
    vector1 = np.random.randint(1, 10, dimension)
    vector2 = np.random.randint(1, 10, dimension)
    print("向量1：", vector1, "向量2：", vector2)
    print("向量加和：", (vector1 + vector2))
    print("向量的内积：", np.dot(vector1, vector2))
    dot = 0
    for i in range(dimension):
        dot += vector1[i] * vector2[i]

    print("向量的内积(公式计算)：", dot)
    print("向量的模：", np.linalg.norm(vector1))
    norm = 0
    for element in vector1:
        norm += element ** 2

    print("向量的模(公式计算)：", np.sqrt(norm))

    # 向量的夹角余弦值，向量的内积/（向量模的乘积）
    cos_data = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    print("向量的夹角余弦值：", cos_data)


if __name__ == '__main__':
    '向量定义'
    # generate_vector()
    '向量运算'
    # vector_operation()