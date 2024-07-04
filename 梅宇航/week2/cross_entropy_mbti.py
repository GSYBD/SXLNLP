import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 定义MBTI类型
mbti_types = [
    'ISTJ', 'ISFJ', 'INFJ', 'INTJ',
    'ISTP', 'ISFP', 'INFP', 'INTP',
    'ESTP', 'ESFP', 'ENFP', 'ENTP',
    'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
]

# 生成模拟MBTI数据
def generate_mbti_data(num_samples, input_dim, num_classes):
    print("生成模拟MBTI数据...")
    np.random.seed(42)
    extraversion = np.random.randint(0, 101, size=(num_samples, 1))  # 外向 (E) - 内向 (I)
    sensing = np.random.randint(0, 101, size=(num_samples, 1))  # 感觉 (S) - 直觉 (N)
    thinking = np.random.randint(0, 101, size=(num_samples, 1))  # 思考 (T) - 情感 (F)
    judging = np.random.randint(0, 101, size=(num_samples, 1))  # 判断 (J) - 知觉 (P)
    
    # 归一化到0到1
    X = np.hstack((extraversion, sensing, thinking, judging)) / 100.0
    
    # 根据评分生成标签：0-50 对应 E/S/T/J，51-100 对应 I/N/F/P
    y = ((extraversion > 50).astype(int) * 8 +
         (sensing > 50).astype(int) * 4 +
         (thinking > 50).astype(int) * 2 +
         (judging > 50).astype(int))
    
    # 打印前20个样本和标签
    for i in range(num_samples):
        extraversion_label = 'I' if extraversion[i] > 50 else 'E'
        sensing_label = 'N' if sensing[i] > 50 else 'S'
        thinking_label = 'F' if thinking[i] > 50 else 'T'
        judging_label = 'P' if judging[i] > 50 else 'J'
        mbti_label = f"{extraversion_label}{sensing_label}{thinking_label}{judging_label}"
        label_index = mbti_types.index(mbti_label)
        print(f"样本 {i + 1}: 特征 = {X[i]}, 标签 = {label_index} ({mbti_label})")
        y[i] = label_index
    
    y_onehot = np.eye(num_classes)[y.reshape(-1)]
    
    return X, y_onehot, y

# 数据准备
num_samples = 10000  # 样本数量
input_dim = 4       # 每个样本的特征维度
num_classes = len(mbti_types)  # MBTI类型数量

# 生成训练数据
X, y_onehot, y = generate_mbti_data(num_samples, input_dim, num_classes)
print(f"生成了{num_samples}个样本，每个样本有{input_dim}个特征。")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_onehot_tensor = torch.tensor(y_onehot, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long).reshape(-1)

dataset = TensorDataset(X_tensor, y_onehot_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型
class MBTIModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MBTIModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# 自定义交叉熵损失函数
class CategoricalCrossentropy(nn.Module):
    def __init__(self):
        super(CategoricalCrossentropy, self).__init__()
    
    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        loss_value = -torch.sum(y_true * torch.log(y_pred + 1e-7)) / batch_size
        return loss_value

# 模型实例化、损失函数和优化器定义
model = MBTIModel(input_dim, num_classes)
criterion = CategoricalCrossentropy()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50  # 增加训练的epoch数量
print("\n开始训练模型...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (X_batch, y_onehot_batch, y_batch) in enumerate(dataloader):
        # 前向传播
        outputs = model(X_batch)
        # 计算损失
        loss = criterion(outputs, y_onehot_batch)
        epoch_loss += loss.item()
        
        # 打印部分损失值
        if batch_idx % 50 == 0:
            print(f'过程交叉熵损失值: {loss.item():.4f}')
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'学习[{epoch+1}/{num_epochs}]次, 平均损失: {epoch_loss / len(dataloader):.4f}')

# 评估模型
print("\n评估模型...")
X_test, y_test_onehot, y_test = generate_mbti_data(2000, input_dim, num_classes)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).reshape(-1)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"测试集上的预测结果: {predicted}")
    print(f"真实标签: {y_test_tensor}")
    print(f'测试准确率: {accuracy:.4f}')

# 用户输入测试
def get_user_input():
    print("\n请输入您的测试数据（0到100的分数）")
    user_data = []
    features = [
        "外向 (E) - 内向 (I)", 
        "感觉 (S) - 直觉 (N)", 
        "思考 (T) - 情感 (F)", 
        "判断 (J) - 知觉 (P)"
    ]
    for feature in features:
        value = float(input(f"{feature} (0 到 100): "))
        user_data.append(value / 100.0)  # 归一化
    return np.array(user_data)

print("\n现在可以输入您的测试数据来预测MBTI类型。")
while True:
    user_data = get_user_input()
    user_data_tensor = torch.tensor(user_data, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        user_output = model(user_data_tensor)
        print(f"模型输出的概率分布: {user_output}")
        _, user_predicted = torch.max(user_output, 1)
        mbti_type = mbti_types[user_predicted.item()]
        print(f"你的MBTI人格是：{mbti_type}！ (MBTI类型索引: {user_predicted.item()})")
    
    cont = input("您想要输入其他数据吗? (yes/no): ")
    if cont.lower() != 'yes':
        break
