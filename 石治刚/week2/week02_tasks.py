import numpy as np

# 定义softmax函数
def softmax(matrix):
    exp_matrix = np.exp(matrix)
    return exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)

# 将目标转化为one-hot编码
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i, t] = 1
    return one_hot_target

# 手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred_prob = softmax(pred)
    target_one_hot = to_one_hot(target, pred_prob.shape)
    
    # 计算交叉熵损失
    epsilon = 1e-12  # 加上一个极小值以防止log(0)
    cross_entropy_loss = -np.sum(target_one_hot * np.log(pred_prob + epsilon)) / batch_size
    
    return cross_entropy_loss

# 示例数据
pred = np.array([[0.3, 0.1, 0.3],
                 [0.9, 0.2, 0.9],
                 [0.5, 0.4, 0.2]])
target = np.array([1, 2, 0])

# 计算交叉熵损失
loss = cross_entropy(pred, target)
print(f"手动实现交叉熵损失：{loss}")
