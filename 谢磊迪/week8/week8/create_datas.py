import numpy as np
import json
import random
import torch
from config import config
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
class  create_datas:
    def __init__(self,config,type1=None):
        self.type1 = type1
        self.sentence_len = config['sentence_len']
        self.train_path =config['train_path']
        self.vocab_path =config['vocab_path']
        self.one_zero_rate = config['one_zero_rate']
        self.all_train_data_size = config['all_train_data_size']
        self.schema_path =config['schema_path']
        self.valid_path =config['valid_path']
        self.create_test_datas()
        self.tran_sentence_int()
        self.mode_type =config['mode_type']

        # if  type1 =='train':
        #     self.create_train_datas()
        # elif type1 =='test':
        #
    def __len__(self):
        if self.type1 == 'train':

            return  config['all_train_data_size']

        elif self.type1 == 'test':
            return  len(self.test_int_li)

    def __getitem__(self,index):
        if self.type1 == 'train':
            return self.create_train_datas()
        elif  self.type1 =='test':
            return self.test_int_li[index]

    def word2int(self):
        word_int_dic ={}
        with  open(self.vocab_path,'r',encoding='utf8', errors='ignore') as fq:
            for id1,word in enumerate(fq):
                word_int_dic[word.strip()]=id1
        return word_int_dic

    def tran_sentence_int(self):
        """
        训练数据每个字映射到word2int中去
        :return:
        """
        self.sentens_li =[]
        self.sentens2schema = defaultdict(list)
        schema_class = self.create_schema_class()
        word_int_dic  = self.word2int()
        with open(self.train_path,'r',encoding='utf8', errors='ignore') as fq:
            for lines in fq:
                questions= json.loads(lines)['questions']
                target = json.loads(lines)['target']
                words_li =[]
                for question in questions:
                    word_li=[0]*self.sentence_len
                    for ind1,word in enumerate(list(question)):
                        if ind1<self.sentence_len:
                            word_li[ind1] = word_int_dic.get(word,word_int_dic['[UNK]'])
                            word_li = torch.LongTensor(word_li)
                    words_li.append(word_li)
                self.sentens2schema[schema_class[target]]=words_li
                self.sentens_li.append(words_li)
    def create_schema_class(self):
        with open(self.schema_path, 'r', encoding='utf8', errors='ignore') as fq:
            schema_class = json.load(fq)
        return schema_class

    def create_test_datas(self):
        word_int_dic = self.word2int()
        schema_class = self.create_schema_class()
        test_li = []
        self.test_int_li = []
        with open(self.valid_path,'r',encoding='utf-8', errors='ignore') as fq:
            for lines in fq:
                test_li.append(json.loads(lines))
            for _test in test_li:
                word_li = [0] * self.sentence_len
                for ind, word in enumerate(_test[0]):
                    if ind < self.sentence_len:
                        word_li[ind] = word_int_dic.get(word, word_int_dic['[UNK]'])
                word_li = torch.LongTensor(word_li)
                self.test_int_li.append([word_li, schema_class.get(_test[1])])
        return

    def create_train_datas(self):
        self.tran_sentence_int()
        tran_data=[]
        if self.mode_type=='Twins':
            #正样本
            if np.random.random() >self.one_zero_rate:
                sentens_class = random.sample(self.sentens_li,1)[0]
                if len(sentens_class)<2:
                    return  self.create_train_datas()
                else:
                    tran_data.extend(random.sample(sentens_class,2))
                    tran_data.append(torch.LongTensor([1]))
                    return tran_data
            #负样本
            else:
                sentens_class1,sentens_class2 = random.sample(self.sentens_li,2)
                tran_data = [random.sample(sentens_class1,1)[0]
                ,random.sample(sentens_class2,1)[0]
                ,torch.LongTensor([-1])]
                return tran_data
        elif  self.mode_type=='Triplet':
            _sentens_li  = self.sentens_li.copy()
            sentens_class = random.sample(_sentens_li, 1)[0]
            if len(sentens_class) < 2:
                return self.create_train_datas()
            else:
                _sentens_li = [x for x in _sentens_li if id(x) != id(sentens_class)]
                sentens_class2 = random.sample(_sentens_li, 1)[0]
                tran_data.extend(random.sample(sentens_class, 2))
                tran_data.extend(random.sample(sentens_class2, 1))
                return tran_data


def load_data(config,type1,shuffle=True):
    dg = create_datas(config,type1)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle,num_workers=4)
    return dl
if __name__ == '__main__':
    # config1 =config
    df = load_data(config,shuffle=True,type1='train')
    # dg = create_datas(config, type1='train')
    # dg.create_train_datas()
    print(1)
    for i in df:
        print(i)
        break






