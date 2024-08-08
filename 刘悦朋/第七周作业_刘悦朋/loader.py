from config import Config
import pandas
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

"""

    数据加载

"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        # 数据集文件路径
        self.data_path = data_path
        if self.config['model_type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
        # 构建字典
        self.vocab = load_vocab(config['vocab_path'])
        # 统计字典长度
        self.config['vocab_size'] = len(self.vocab)
        # 整理样本, 将字符串转变为idx列表, 以向量形式存入self.data列表
        self.data = []
        self.load()

    def load(self):
        """
        将csv文件中的数据, label review,
        转化为idx列表, [[idx, idx, ..., idx], [label]]
        存入self.data列表, [[[idx, idx, ..., idx], [label]], ..., [[idx, idx, ..., idx], [label]]]
        """
        df = pandas.read_csv(self.data_path)
        labels = df['label']
        self.config['class_num'] = len(set(list(labels)))
        reviews = df['review']
        if self.config['model_type'] == 'bert':
            for label, review in zip(labels, reviews):
                sen_to_idx = self.tokenizer.encode(review, max_length=self.config['max_length'], padding='max_length',
                                                   truncation=True)
                self.data.append([torch.LongTensor(sen_to_idx), torch.FloatTensor([label])])
        else:
            for label, review in zip(labels, reviews):
                self.data.append([torch.LongTensor(self.encode_sentence(review)), torch.FloatTensor([label])])

    def encode_sentence(self, sentence):
        """
        将输入文本转化为idx列表, [idx, idx, idx, ..., idx]
        """
        sentence_to_idx = []
        for char in sentence:
            sentence_to_idx.append(self.vocab.get(char, self.vocab['[UNK]']))
        sentence_to_idx = self.padding(sentence_to_idx)
        return sentence_to_idx

    def padding(self, sentence_to_idx):
        sentence_to_idx = sentence_to_idx[:self.config['max_length']]
        sentence_to_idx += [0] * (self.config['max_length'] - len(sentence_to_idx))
        return sentence_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    """
    将每行一个字符的字表txt文件,
    转化为字典格式
    """
    token_dict = {}
    with open(vocab_path, encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置, 所以从1开始
    return token_dict


def load_data(data_path, config, shuffle=True):
    """
    生成DataLoader类,
    通过enumerate(DataLoader)遍历,
    长度为batch数量,
    每一个batch分为sample_batch及label_batch,
    DataLoader.dataset为DataGenerator类,
    DataGenerator.data为列表, 可以查看原来的样本, 即[[[idx, idx, ..., idx], [label]], ..., [[idx, idx, ..., idx], [label]]]
    """
    data_generator = DataGenerator(data_path, config)
    data_loader = DataLoader(data_generator, batch_size=config['batch_size'], shuffle=shuffle)
    return data_loader


if __name__ == '__main__':
    # 生成DataLoader类
    train_data = load_data('文本分类练习.csv', Config)
    # 统计样本数量
    print('Found %d samples belonging to %d classes.\n' % (len(train_data.dataset), Config['class_num']))
    # 展示第一个batch中前三条样本
    train_data_iter = iter(train_data)
    review_batch, label_batch = next(train_data_iter)
    for i in range(3):
        print('Review', review_batch[i])
        print('Label', label_batch[i], '\n')
