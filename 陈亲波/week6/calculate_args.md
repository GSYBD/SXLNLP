# 计算Bert 中可训练参数数量
> 以 Bert-Base 模型配置为例
## 1. 输入嵌入层 embedding layer
假设词汇表大小V = 30000
每个词向量维度为d_model = 768
### 1. 词嵌入矩阵
word Embeddings = V * d_model = 30000 * 768 = 23040000
### 2. 位置嵌入矩阵
position Embeddings = L * d_model = 512 * 768 = 393216

### 3. 分段嵌入矩阵
假设分段数量S=2, 每个向量维度d_model = 768,
Segment Embeddings = S *d_model = 2 * 768 =1536

### 4. 总量
Total = 23040000+393216+1536 = 23434752
## 2. 自注意力层 Self-attention Layer

### 1. 线性变换
每个注意力头有3个线性变换矩阵Query,Key,Value. 每个矩阵的维度为 d_model *d_k,
d_k = d_model/h = 768/12 =64

每个注意力头的参数量为：
parameters per head = 3 * d_model *d_k = 3*768*64 = 147456

每层注意力头的总参数量为：
Attention Parameters per layer = 12 * 147456 = 1769472

每层输出线性变换后的矩阵维度为 d_model *d_model
Output = 768 *768 = 589824

自注意力层总参数量
Total = Attention + Linear = 1769472 + 589824 


## 3. 前馈神经网络
### 1. 第一个线性变换
d_model = 768

d_ff = 4*d_model = 3072

First_transformation = d_model * d_ff = 768 * 3072 = 2359296

Second_transformation = d_ff *d_model = 3072 *768 = 2359296

每层的前馈神经网络总数为
total = First_transformation+Second_transformation = 4718592

## 4. 总参数量计算

total of 12 layers = 12 *(2359296+4718592) = 84934656

total = 23434752 +84934656 = 108369408


