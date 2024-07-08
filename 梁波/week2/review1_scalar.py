import numpy as np

'''
一、标量（Scalar）
    标量就是一个单独的数
'''


def generate_scalar():
    # 生成[0,1)随机数，即标量
    scalar1 = np.random.rand()
    print(scalar1)

    # 生成[1,10]随机数
    scalar2 = np.random.uniform(1, 10)
    print(scalar2)

    # 生成[1,10)随机整数
    scalar3 = np.random.randint(1, 10)
    print(scalar3)


if __name__ == '__main__':
    '标量'
    # generate_scalar()


