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

max_len = 128
batch_size = 16
learning_rate = 1e-4
embed_dim = 128
hidden_dim = 128
num_classes = 2

data = pd.read_csv('/Users/lishuyan/PycharmProjects/lsy823/SXLNLP/李姝妍/week7/文本分类练习.csv')

comment = data["review"].tolist()
labels = data["label"].tolist()

combined = list(zip(comment, labels))
random.shuffle(combined)
comment[:], labels[:] = zip(*combined)

train_comment, valid_comment, train_labels, valid_labels = train_test_split(comment, labels, test_size=0.2,
                                                                            random_state=42)

class CustomDataset(Dataset):
    def __init__(self, comment, labels, tokenizer, max_len):
        self.comment = comment
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comment)

    def __getitem__(self, item):
        review = str(self.comment[item])
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


def create_data_loader(comment, labels, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        comment=comment,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )


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


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data_loader = create_data_loader(train_comment, train_labels, tokenizer, max_len, batch_size)
valid_data_loader = create_data_loader(valid_comment, valid_labels, tokenizer, max_len, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = pd.DataFrame(columns=["Model", "Learning_rate", "Hidden_size", "Batch_size",
                                "Train_Avg_Positive_sample", "Train_Avg_Negative_sample",
                                "Valid_Avg_Positive_sample", "Valid_Avg_Negative_sample",
                                "Train_Avg_length", "Valid_Avg_length",
                                "Last_Accuracy", "time"])

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
    "Train_Avg_length": np.mean([len(r.split()) for r in train_comment]),
    "Valid_Avg_length": np.mean([len(r.split()) for r in valid_comment]),
    "Last_Accuracy": accuracy,
    "time": training_time
}, ignore_index=True)

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
        "Train_Avg_length": np.mean([len(r.split()) for r in train_comment]),
        "Valid_Avg_length": np.mean([len(r.split()) for r in valid_comment]),
        "Last_Accuracy": accuracy,
        "time": training_time
    }, ignore_index=True)

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
        "Train_Avg_length": np.mean([len(r.split()) for r in train_comment]),
        "Valid_Avg_length": np.mean([len(r.split()) for r in valid_comment]),
        "Last_Accuracy": accuracy,
        "time": training_time
    }, ignore_index=True)

    print(results)
