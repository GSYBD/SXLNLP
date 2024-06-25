"""文本序列化将长短不一致的句子转化为数字"""

import numpy as np

class word2seq():
    UNK_TAG="UNK"
    PAD_TAG="PAD"
    UNK=0
    PAD=1
    def __init__(self):
        # 定义一个词典
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.count={}#统计词频

    def __len__(self):
        return len(self.dict)


    def fit(self,sentence):
        """
        :param sentence:[word,word...]
        """
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1

    def build_vocab(self,min=None,max=None,max_features=None):
        """生成词表
        :param min:
        :param max:
        """
        if min is not None:
            self.count = {word:value  for word ,value in self.count.items() if value >min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        if max_features is not None:#限制保留词语数
            temp = sorted(self.count.items(),key=lambda x:x[-1] ,reverse=True)[:max_features]
            self.count = dict(temp)
        for word in self.count:
            self.dict[word] = len(self.dict)
        #句子进行文本转化成数据
        #反转的dict字典
        self.in_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len=None):
        "把句子转为序列"
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len - len(sentence))#填充
            if max_len < len(sentence):
                sentence = sentence[:max_len]#裁剪
        #for word in  sentence:
           # self.dict.get(word,self.UNK)
        return [self.dict.get(word,self.UNK) for word  in  sentence]


    def in_transform(self,indices):#将序列转化成文本
        return [self.in_dict.get(idx) for idx in indices]

if __name__ == '__main__':
    #    ws = word2seq()
    #    ws.fit(["我","我","是","谁","我"])
    #    ws.build_vocab()
    #    print(ws.dict)
    #    ret = ws.transform(["我","我","是","谁","我"],max_len=6)
    #    print(ret)
    #    concat = ws.in_transform(ret)
    #    print(concat)

    from dataset01 import token
    from word2seq import word2seq
    import pickle
    ws = word2seq()
    data_path = r"D:\NLP\数据集\sms+spam+collection\SMSSpamCollection"
    lines = open(data_path, encoding='gb18030', errors='ignore').readlines()
    con = []
    for line in lines:
        con1 = token(line[4:].strip())
        ws.fit(con1)
        con = con + con1
        # for word in word.split():
        #     cur_lines.append(word)
        #     con = token(cur_lines)
    ws.build_vocab(min=3)
    pickle.dump(ws,open("./ws.pkl", "wb"))
    print(len(ws))
