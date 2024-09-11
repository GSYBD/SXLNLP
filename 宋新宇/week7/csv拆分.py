import pandas as pd
from sklearn.model_selection import train_test_split

# 步骤1: 读取CSV文件
data = pd.read_csv(r"E:\八斗学院AI课程\NLP\八斗精品班\第七周 文本分类问题\week7 文本分类问题\week7 文本分类问题\文本分类练习数据集\文本分类练习.csv")

# 步骤2: 提取特征和标签
x = data['review']  # 特征数据：文本评论
y = data['label']  # 目标变量：标签

# 步骤2: 切分数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 将训练集和测试集分别转换为DataFrame，以便保存为CSV
train_data = pd.concat([y_train, x_train], axis=1)
test_data = pd.concat([y_test, x_test], axis=1)

# 打印结果，看看数据切分是否成功
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 步骤3: 存储到CSV文件
train_data.to_csv('train_data.csv', index=False)  # 不保存索引
test_data.to_csv('test_data.csv', index=False)  # 不保存索引

print("数据已成功切分并保存到CSV文件中。")