import json
from config import Config
import os
import torch
from model import TorchModel, choose_optimizer


# 加载模型和tokenizer
class Sentencelabel:
    def __init__(self,config,model_path):
        self.config=config
        self.schema=self.load_schema(config['schema_path'])
        self.index_to_sign=dict((y,x) for x,y in self.schema.items())
        self.vocab=self.load_vocab(config['vocab_path'])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print('模型加载完毕')

    def load_schema(self,path):
        with open(path,encoding='utf8') as f:
            schema=json.load(f)
            self.config['class_num']=len(schema)
        return schema
    def load_vocab(self,path):
        token_dict={}
        with open(path,encoding='utf8') as f:
            for index,line in enumerate(f):
                token=line.strip()
                token_dict[token]=index+1
        self.config['vocab_size']=len(token_dict)
        return token_dict
    def predict(self,sentence):
        input_id=[]
        for char in sentence:
            input_id.append(self.vocab.get(char,self.vocab['[UNK]']))
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            res =torch.argmax(res,dim=1)
        label_senctence = ''
        out_put=set()
        start=0
        end =0
        while start< len(res):
            if (self.index_to_sign[int(res[start])])[0]=='B':
                label_senctence+=sentence[start]
                end=start+1
                while (self.index_to_sign[int(res[end])])[0]=='I':
                    label_senctence += sentence[end]
                    end+=1
                start=end
            else:
                start+=1
            if label_senctence != '':
                out_put.add(label_senctence)
                label_senctence=''
        return out_put

if __name__=='__main__':
    s1=Sentencelabel(Config,'model_output/epoch_20.pth')
    sentence='我决定在9月的某一天去中国的其他地方旅游'
    res=s1.predict(sentence)
    print(res)





