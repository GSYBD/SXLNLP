# -*- coding: utf-8 -*-
import json
import logging
import random

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Dataset():
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class XFileLoader:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    def load_data(self):
        self.data = []
        self.target_data = []
        self.mask = []
        with open(self.config["train_data_path"],"r",encoding="utf8") as f:
            for i,line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                input_ids,target_input_ids,mask= self.prepare_data(title, content)
                self.data.append(input_ids)
                self.target_data.append(target_input_ids)
                self.mask.append(mask)

    #输入输出转化成序列
    def prepare_data(self, title, content):
        max_length = self.config["max_length"]
        # 输入 问题+答案用seq连接
        title_input_ids = self.tokenizer.encode(title, add_special_tokens=False) + [102]
        content_input_ids = self.tokenizer.encode(content,add_special_tokens=False,
                                                  padding='max_length', max_length=max_length - len(title_input_ids), truncation=True)
        input_ids = title_input_ids+content_input_ids

        # 答案
        content_input_ids = self.tokenizer.encode(content,add_special_tokens=False) + [10434] #10434 结束符
        target_input_ids = [-100] * (len(title_input_ids)-1) + content_input_ids
        target_input_ids = self.padding(target_input_ids, max_length)


        # 构建下三角mask
        mask = self.build_mask(len(title_input_ids),len(content_input_ids)-1,max_length)

        return input_ids,target_input_ids,mask
        # return self.data,self.target
    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [0] * (length - len(input_id))
        return input_id

    def build_train_data(self):
        data = []
        for x,y,mask in zip(self.data,self.target_data,self.mask):
            data.append([torch.LongTensor(x), torch.LongTensor(y),mask])
        dl = DataLoader(data, batch_size=self.config["batch_size"], shuffle=True)
        return dl

    def build_mask(self,s1_len,s2_len,max_len):
        z1 = torch.ones((s1_len, s1_len))
        z2 = torch.ones((s2_len, s1_len))
        z3 = torch.zeros((s1_len, s2_len))
        z4 = torch.tril(torch.ones((s2_len, s2_len)))
        z1_z3 = torch.cat((z1, z3), dim=-1)

        z2_z4 = torch.cat((z2, z4), dim=-1)

        z = torch.cat((z1_z3, z2_z4), dim=0)

        pad = torch.nn.ZeroPad2d(padding=(0, max_len - z.shape[-1], 0, max_len - z.shape[-1]))
        z_mask = pad(z)
        return z_mask





if __name__ == '__main__':
    # x, y = XDataLoader("xx").build_sample(10)
    # print(x, '>>>>', y)
    from config import Config

    # dl = XRandomDataLoader(Config).load_data("xx")
    # for d in dl:
    #     print(d)
    #     print(",,,,")
    tokenizer = BertTokenizer.from_pretrained(Config["bert_model_path"])
    xfileLoader = XFileLoader(Config,tokenizer)
    xfileLoader.load_data()
    dl = xfileLoader.build_train_data()
    for d in dl:
        print(d)
        print(",,,,")
