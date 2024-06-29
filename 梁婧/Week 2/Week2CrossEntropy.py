import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

'''
随机将1-9中的某一位替换为0，预测0的位置，按照多分类项目处理，分为9类
'''

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 9)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.linear(x)
        return logits


def build_sample():
    vector = np.arange(1, 10)
    random_index = np.random.randint(0, len(vector))
    vector[random_index] = 0
    return vector, random_index


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 使用训练好的模型做预测
def predict(model_path, test_vecs):
    # 加载模型
    model = TorchModel(input_size=9)
    model.load_state_dict(torch.load(model_path)) # 加载训练好的权重
    print(model.state_dict())
    model.eval()  # 测试模式

    predictions = []

    # 逐个处理向量
    with torch.no_grad():  # 不计算梯度
        for vec in test_vecs:
            vec_tensor = torch.FloatTensor(vec).unsqueeze(0)  # 添加一个批次维度
            logits = model(vec_tensor)
            _, predicted_class = torch.max(logits, 1)
            predictions.append(predicted_class.item())

    return predictions

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    correct = 0
    total = y.size(0)  # y是一个一维张量，包含所有样本的真实标签
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，输出是logits
        # 将logits转换为概率分布
        probs = F.softmax(y_pred, dim=1)
        # 获取最大概率的索引作为预测类别
        _, predicted = torch.max(probs, 1)
        # 比较预测类别和真实标签
        correct = (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print("正确预测个数：%d, 总预测数：%d, 正确率：%.2f%%" % (correct, total, accuracy))
    return accuracy


def main():
    # 配置参数
    epochs_num = 20
    batch_size = 20  # 每次训练样本个数
    train_sample_num = 5000  # 训练样本总数
    input_size = 9  # 输入向量维度
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size)

    # 选择优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练集
    train_x, train_y = build_dataset(train_sample_num)

    # 训练过程
    for epoch in range(epochs_num):
        model.train()
        # 批次循环
        for batch_index in range(train_sample_num // batch_size):
            # 数据加载
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_index]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_index]
            logits = model(x)
            loss = model.loss_fn(logits, y)
            loss.backward()  # 梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 权重归零
        print(f'Epoch [{epoch + 1}/{epochs_num}], Loss: {loss.item():.4f}')
        acc = evaluate(model)  # 测试本轮模型结果
        print(acc)


    #保存模型
    torch.save(model.state_dict(), "model.pt")

# 生成测试数据
def generate_prediction_data(num_samples):
    prediction_data = []

    # 为每个样本生成一个向量
    for i in range(num_samples):
        vector = np.arange(1, 10)
        random_index = np.random.randint(0, len(vector))
        vector[random_index] = 0  # 将随机位置的元素设置为0
        prediction_data.append(vector)

    return prediction_data


if __name__ == "__main__":
    main()
    # 生成5个预测数据
    prediction_vecs = generate_prediction_data(5)
    print(prediction_vecs)
    print('\n---------------------------\n')
    # 调用predict_multiple函数并打印预测结果
    predictions = predict("model.pt", prediction_vecs)
    print(predictions)