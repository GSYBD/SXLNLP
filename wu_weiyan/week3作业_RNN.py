"""

用RNN进行分类任务

"""

import torch
import torch.nn as nn
from  torch.utils.data import Dataset, DataLoader
import string
from io import open

# # 获取所有常用字符包括字母和常用标点
all_letters = string.ascii_letters + " .,;'"
#
# # 获取常用字符数量
n_letters = len(all_letters)
# # print("n_letter:", n_letters)
#
# # 国家名 种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']

# 国家名 个数
categorynum = len(categorys)
# # print('categorys--->', categorys)

def read_data(filename):
    my_list_x, my_list_y= [], []
    # 打开文件
    with  open(filename, mode='r', encoding='utf-8') as f:
        # 按照行读数据
        for line in f.readlines():
            if len(line) <= 5:
                continue
            # 按照行提取样本x 样本y
            (x, y) = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)

    # 返回样本x的列表、样本y的列表
    return my_list_x, my_list_y

class NameClassDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        # 样本x
        self.my_list_x = my_list_x
        # 样本y
        self.my_list_y = my_list_y
        # 样本条目数
        self.sample_len = len(my_list_x)

    # 获取样本条数
    def __len__(self):
        return self.sample_len

    # 获取第几条 样本数据
    def __getitem__(self, index):

        # 对index异常值进行修正 [0, self.sample_len-1]
        index = min(max(index, 0), self.sample_len-1)

        # 按索引获取 数据样本 x y
        x = self.my_list_x[index]
        y = self.my_list_y[index]

        # 样本x one-hot张量化
        tensor_x = torch.zeros(len(x), n_letters)
        # 遍历人名 的 每个字母 做成one-hot编码
        for li, letter in enumerate(x):
            # letter2indx 使用all_letters.find(letter)查找字母在all_letters表中的位置 给one-hot赋值
            tensor_x[li][all_letters.find(letter)] = 1

        # 样本y 张量化
        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)

        # 返回结果
        return tensor_x, tensor_y

# 构建迭代器遍历数据
def dm_test_NameClassDataset():

    # 1 获取数据
    myfilename = './data/name_classfication.txt'
    my_list_x, my_list_y = read_data(myfilename)
    print('my_list_x length', len(my_list_x))
    print('my_list_y length', len(my_list_y))

    # 2 实例化dataset对象
    nameclassdataset = NameClassDataset(my_list_x, my_list_y)

    # 3 实例化dataloader
    mydataloader = DataLoader(dataset=nameclassdataset, batch_size=1, shuffle=True)
    for  i, (x, y) in enumerate (mydataloader):
        print('x.shape', x.shape, x)
        print('y.shape', y.shape, y)
        break

# 构造RNN网络
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        # 1 init函数 准备三个层 self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 定义rnn层
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)

        # 定义linear层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):

        # 数据形状 [6,57] -> [6,1,57]
        input = input.unsqueeze(1)
        # 数据经过模型 提取事物特征
        rr, hn = self.rnn(input, hidden)

        tmprr = rr[-1]

        # 数据经过全连接层 [1,128] -->[1,18]
        tmprr = self.linear(tmprr)

        # 数据经过softmax层返回
        return self.softmax(tmprr), hn

    def inithidden(self):
        # 初始化隐藏层输入数据 inithidden()
        return torch.zeros(self.num_layers, 1,self.hidden_size)

def train():

    # 实例化rnn对象
    myrnn = RNN(57, 128, 18)

    # 准备数据
    input = torch.randn(6, 57)

    hidden = myrnn.inithidden()

    # 给模型1次性的送数据
    output, hidden = myrnn(input, hidden)
    print('一次性的送数据：output->\n', output.shape, output)
    print('hidden->', hidden.shape)

    # 给模型1个字符1个字符的喂数据
    hidden = myrnn.inithidden()
    for i in range(input.shape[0]):
        tmpinput = input[i].unsqueeze(0)
        output, hidden = myrnn(tmpinput, hidden)

    # 最后一次ouput
    print('一个字符一个字符的送数据output->\n', output.shape, output)

if __name__ == '__main__':
    train()