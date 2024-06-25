"""处理骚扰短信对应的数据集"""

from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
import numpy as np
import torch
import re
from lib import ws ,max_len


data_path = r"D:\NLP\数据集\sms+spam+collection\SMSSpamCollection"
eval_path = r"D:\NLP\数据集\sms+spam+collection\evaldataset.txt"

def token(con):
#    con = re.sub("？"," ",con)
    filters = ['!',  '#', '96',  '%', '\t','\.','\?']
    con = re.sub("|".join(filters)," ", con)
    token =[i.strip().lower()  for i in con.split()]
    return token


class mydataset (Dataset):
    def __init__(self):
        self.lines = open(data_path,encoding='gb18030' , errors='ignore').readlines()


    def __getitem__(self, index):
        cur_lines = self.lines[index].strip()
        laber = cur_lines[:4].strip()
        laber = 0 if laber =="spam" else 1
        con = token(cur_lines[4:].strip())
        #con = torch.LongTensor([con])
        #laber = torch.LongTensor([laber])
        return laber ,con

    def __len__(self):
        return len(self.lines)

class eval_dataset (Dataset):
    def __init__(self):
        self.lines = open(eval_path,encoding='gb18030' , errors='ignore').readlines()

    def __getitem__(self, index):
        cur_lines = self.lines[index].strip()
        laber = cur_lines[:4].strip()
        laber = 0 if laber =="spam" else 1
        con = token(cur_lines[4:].strip())
        #con = torch.LongTensor([con])
        #laber = torch.LongTensor([laber])
        return laber ,con

    def __len__(self):
        return len(self.lines)


def collate(batch):
    # laber = [i[0] for i in batch]
    # con = [i[1] for i in batch]
    # laber = np.array(laber)
    # con = np.array(con)
    laber,con = list(zip(*batch))
    con = [ws.transform(i,max_len=max_len)for i in con]
    laber = torch.LongTensor(laber)
    con = torch.LongTensor(con)
    return laber,con

mydataset = mydataset()
mydataloader = DataLoader(mydataset,batch_size=60,shuffle=True,drop_last=True,collate_fn =collate)
eval_dataset = eval_dataset()
eval_dataloader = DataLoader(eval_dataset,batch_size=2,shuffle=True,drop_last=True,collate_fn =collate)


if __name__ == '__main__':
   for idx,(laber,con) in enumerate(mydataloader):
        print(laber)
        print(con)
        break
#   con = "#|| i'll be home in a ! ? few weeks anyway. || what are the plans"
#   print(con)
#   print(token(con))
#   print(mydataset[2])