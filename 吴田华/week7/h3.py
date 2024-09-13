import pandas as pd
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset

# 加载数据
data = pd.read_csv('/mnt/data/文本分类练习.csv')

# 假设评论数据在 "review" 列，标签在 "label" 列
reviews = data["review"].tolist()
labels = data["label"].tolist()

# 打乱数据
combined = list(zip(reviews, labels))
random.shuffle(combined)
reviews[:], labels[:] = zip(*combined)

# 按8:2分割数据集
train_reviews, valid_reviews, train_labels, valid_labels = train_test_split(reviews, labels, test_size=0.2,
                                                                            random_state=42)

# 打印训练和验证集的一些统计信息
print(f"训练集大小: {len(train_reviews)}, 验证集大小: {len(valid_reviews)}")
print(f"训练集中正样本数: {sum(train_labels)}, 负样本数: {len(train_labels) - sum(train_labels)}")
print(f"验证集中正样本数: {sum(valid_labels)}, 负样本数: {len(valid_labels) - sum(valid_labels)}")


class CustomDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(reviews, labels, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        reviews=reviews,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


# 定义简单的CNN模型
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv2d(1, 128, (5, embed_dim))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((2, 1))
        self.fc = nn.Linear(128 * (max_len // 2 - 2), num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 定义简单的RNN模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.rnn(x)
        x = self.fc(h_n[-1])
        return x


# 评估模型函数
def evaluate(model, data_loader, device):
    model = model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            true_labels.extend(labels)

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


# 超参数设置
max_len = 128
batch_size = 16
learning_rate = 1e-4
embed_dim = 128
hidden_dim = 128
num_classes = 2

# 创建数据加载器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data_loader = create_data_loader(train_reviews, train_labels, tokenizer, max_len, batch_size)
valid_data_loader = create_data_loader(valid_reviews, valid_labels, tokenizer, max_len, batch_size)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 结果表格初始化
results = pd.DataFrame(columns=["Model", "Learning_rate", "Hidden_size", "Batch_size",
                                "Train_Avg_Positive_sample", "Train_Avg_Negative_sample",
                                "Valid_Avg_Positive_sample", "Valid_Avg_Negative_sample",
                                "Train_Avg_length", "Valid_Avg_length",
                                "Last_Accuracy", "time"])

# 训练和评估BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_loader.dataset,
    eval_dataset=valid_data_loader.dataset
)
start_time = time.time()
trainer.train()
training_time = time.time() - start_time
eval_results = trainer.evaluate()
accuracy = eval_results['eval_accuracy']
results = results.append({
    "Model": "BERT",
    "Learning_rate": training_args.learning_rate,
    "Hidden_size": model.config.hidden_size,
    "Batch_size": training_args.per_device_train_batch_size,
    "Train_Avg_Positive_sample": sum(train_labels) / len(train_labels),
    "Train_Avg_Negative_sample": (len(train_labels) - sum(train_labels)) / len(train_labels),
    "Valid_Avg_Positive_sample": sum(valid_labels) / len(valid_labels),
    "Valid_Avg_Negative_sample": (len(valid_labels) - sum(valid_labels)) / len(valid_labels),
    "Train_Avg_length": np.mean([len(r.split()) for r in train_reviews]),
    "Valid_Avg_length": np.mean([len(r.split()) for r in valid_reviews]),
    "Last_Accuracy": accuracy,
    "time": training_time
}, ignore_index=True)

# 训练和评估CNN模型
cnn_model = CNNModel(len(tokenizer.vocab), embed_dim, num_classes).to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
start_time = time.time()
for epoch in range(3):
    cnn_model.train()
    for batch in train_data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = cnn_model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    training_time = time.time() - start_time
    accuracy = evaluate(cnn_model, valid_data_loader, device)
    results = results.append({
        "Model": "CNN",
        "Learning_rate": learning_rate,
        "Hidden_size": embed_dim,
        "Batch_size": batch_size,
        "Train_Avg_Positive_sample": sum(train_labels) / len(train_labels),
        "Train_Avg_Negative_sample": (len(train_labels) - sum(train_labels)) / len(train_labels),
        "Valid_Avg_Positive_sample": sum(valid_labels) / len(valid_labels),
        "Valid_Avg_Negative_sample": (len(valid_labels) - sum(valid_labels)) / len(valid_labels),
        "Train_Avg_length": np.mean([len(r.split()) for r in train_reviews]),
        "Valid_Avg_length": np.mean([len(r.split()) for r in valid_reviews]),
        "Last_Accuracy": accuracy,
        "time": training_time
    }, ignore_index=True)

    # 训练和评估RNN模型
    rnn_model = RNNModel(len(tokenizer.vocab), embed_dim, hidden_dim, num_classes).to(device)
    optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(3):
        rnn_model.train()
        for batch in train_data_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = rnn_model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    accuracy = evaluate(rnn_model, valid_data_loader, device)
    results = results.append({
        "Model": "RNN",
        "Learning_rate": learning_rate,
        "Hidden_size": hidden_dim,
        "Batch_size": batch_size,
        "Train_Avg_Positive_sample": sum(train_labels) / len(train_labels),
        "Train_Avg_Negative_sample": (len(train_labels) - sum(train_labels)) / len(train_labels),
        "Valid_Avg_Positive_sample": sum(valid_labels) / len(valid_labels),
        "Valid_Avg_Negative_sample": (len(valid_labels) - sum(valid_labels)) / len(valid_labels),
        "Train_Avg_length": np.mean([len(r.split()) for r in train_reviews]),
        "Valid_Avg_length": np.mean([len(r.split()) for r in valid_reviews]),
        "Last_Accuracy": accuracy,
        "time": training_time
    }, ignore_index=True)

    # 打印最终结果表格
    print(results)

    # 保存结果到csv文件
    results.to_csv('/mnt/data/model_comparison_results.csv', index=False)


