import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 假设参数
vocab_size = 10000  # 词汇表大小
max_length = 100  # 文本最大长度
embedding_dim = 128  # 词嵌入维度
lstm_units = 64  # LSTM单元数
num_classes = 4  # 类别数

# 模拟数据
texts = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
    [2, 22, 32, 42, 52, 62, 72, 82, 92, 100],
    [3, 33, 43, 53, 63, 73, 83, 93, 1, 2]
]
labels = [0, 1, 2, 3, 0]

# 填充序列到相同长度
texts = pad_sequences(texts, maxlen=max_length, padding='post')

# 将标签转换为one-hot编码
labels = to_categorical(labels, num_classes=num_classes)

# 构建模型
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(lstm_units),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 模型摘要
model.summary()

# 训练模型
model.fit(texts, labels, epochs=10, batch_size=2)

# 略有借鉴
