import numpy as np

from create_datas import create_datas
from config import config
import  torch
from create_datas import create_datas,load_data
class Predict():
    def __init__(self,model_path,cpnfig):
        self.model= torch.load(model_path)
        self.config=cpnfig
        self.sentence_len = self.config['sentence_len']
        self.train_datas_vector()

    def train_datas_vector(self):
        train_data = create_datas(self.config, type1='train')
        schema_class = train_data.create_schema_class()
        self.sentence_li = []
        self.schema_li = []
        for schema, sentences in train_data.sentens2schema.items():
            for sentence in sentences:
                self.sentence_li.append(sentence)
                self.schema_li.append(schema)
        with torch.no_grad():
            _sentence_li = torch.stack(self.sentence_li, dim=0)
            if torch.cuda.is_available():
                _sentence_li = _sentence_li.cuda()
            _sentence_vector = self.model(_sentence_li)
            # 将所有向量都作归一化 v / |v|
            self._sentence_vector = torch.nn.functional.normalize(_sentence_vector, dim=-1)

    def predict_datas(self,sentens):
        if isinstance(sentens, str):
            type1='None'
            sentence_len = config['sentence_len']
            word_int_dic = create_datas(config,type1).word2int()
            word_li = [0] * self.sentence_len
            for ind, word in enumerate(sentens):
                if ind < self.sentence_len:
                    word_li[ind] = word_int_dic.get(word, word_int_dic['[UNK]'])
            word_li = self.model(torch.LongTensor(word_li).unsqueeze(0))
            return word_li
        elif isinstance(sentens,list):
            all_word_li=[]
            for i in sentens:
                word_li = self.predict_datas(i)
                all_word_li.append(word_li)
            all_word_li= torch.stack(all_word_li, dim=0)
            return   all_word_li

    def predict(self,sentens):
        all_word_li = self.predict_datas(sentens)
        for i  in all_word_li:
            res = torch.mm(i.unsqueeze(0),self._sentence_vector.T)
            ind = res.squeeze().argmax()
            return self.schema_li[ind]


if __name__=='__main__':
    model_path=r'./model_output/model3.pt'
    pred = Predict(model_path,config)
    print(pred.predict(["密码不记得了"]))


