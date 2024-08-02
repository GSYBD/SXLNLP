# main.py
import torch
import random
import numpy as np
import logging
import os
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from loader import TextDataset, create_vocab, load_vocab
from model import TorchModel
from evaluate import Evaluator
from config import Config

# 设置随机种子
seed = Config.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载数据并分离评论和标签
file_path = Config.train_data_path
data = pd.read_csv(file_path, header=None, names=['label', 'text'])
texts = data['text'].tolist()
labels = data['label'].tolist()

# 打乱数据
texts, labels = shuffle(texts, labels, random_state=42)

# 按8:2分割数据集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 打印一些统计信息
train_lengths = [len(text) for text in train_texts]
test_lengths = [len(text) for text in test_texts]
print(f'训练集样本数: {len(train_texts)}')
print(f'测试集样本数: {len(test_texts)}')
print(f'训练集平均文本长度: {np.mean(train_lengths)}')
print(f'测试集平均文本长度: {np.mean(test_lengths)}')
print(f'训练集文本最大长度: {np.max(train_lengths)}')
print(f'测试集文本最大长度: {np.max(test_lengths)}')

# 训练主函数
def main(config):
    if not os.path.isdir(config.model_path):
        os.mkdir(config.model_path)

    # 加载数据
    vocab = load_vocab(config.vocab_path)
    train_dataset = TextDataset(train_texts, train_labels, vocab, max_length=config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataset = TextDataset(test_texts, test_labels, vocab, max_length=config.max_length)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    # 加载模型
    model = TorchModel(config)
    if torch.cuda.is_available():
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练和评估
    for epoch in range(config.epoch):
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, (input_ids, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                input_ids, labels = input_ids.cuda(), labels.cuda()

            optimizer.zero_grad()
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_loader) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))

        # 评估模型
        evaluator = Evaluator(config, model, logger)
        acc = evaluator.eval(epoch)
    return acc

if __name__ == "__main__":
    headers = ["模型", "学习率", "hidden_size", "batch_size", "pooling_style", "准确率"]
    results_path = 'D:/BaiduNetdiskDownload/week7 文本分类问题/文本分类练习数据集/output.csv'

    with open(results_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for model in ["cnn", "lstm", "gru"]:
            Config.model_type = model
            for lr in [1e-3, 1e-4]:
                Config.learning_rate = lr
                for hidden_size in [128]:
                    Config.hidden_size = hidden_size
                    for batch_size in [32, 64]:
                        Config.batch_size = batch_size
                        for pooling_style in ["avg", "max"]:
                            Config.pooling_style = pooling_style
                            accuracy = main(Config)
                            writer.writerow([model, lr, hidden_size, batch_size, pooling_style, accuracy])
