import torch

from create_datas  import  load_data,create_datas
from config import config

class  Evaluator():
    def __init__(self,config,model):
        self.model=model
        self.train_datas = load_data(config,shuffle=True,type1='train')
        self.test_datas = load_data(config,shuffle=False,type1='test')
        # self.train_datas_vector()

    def train_datas_vector(self):
        train_data = create_datas(config, type1='train')
        schema_class = train_data.create_schema_class()
        self.sentence_li = []
        self.schema_li =[]
        for schema,sentences in train_data.sentens2schema.items():
            for sentence in sentences:
                self.sentence_li.append(sentence)
                self.schema_li.append(schema)
        with torch.no_grad():
            _sentence_li = torch.stack(self.sentence_li, dim=0)
            if torch.cuda.is_available():
                _sentence_li = _sentence_li.cuda()
            _sentence_vector  = self.model(_sentence_li)
            # 将所有向量都作归一化 v / |v|
            self._sentence_vector = torch.nn.functional.normalize(_sentence_vector, dim=-1)

    def eval(self,epoch):
        print(f'第{epoch}轮测试')
        self.model.eval()
        self.right_count = 0
        self.error_count = 0
        for ind,sentens_tests in enumerate(self.test_datas):
            sentens_test,label_test = sentens_tests
            with torch.no_grad():
                _sentens_test = self.model(sentens_test)

            self.duibi(_sentens_test,label_test)
        all_count = self.right_count+self.error_count
        rate = round(self.right_count/all_count*100,2)
        print(f"准确率是:{rate}")

    def duibi(self,_sentens_test,label_test):
        assert  len(_sentens_test)==len(label_test)
        self.train_datas_vector()
        for  _senten_test,label in zip(_sentens_test,label_test):
            res = torch.mm(_senten_test.unsqueeze(0),self._sentence_vector.T)
            idn1= res.squeeze().argmax()
            pred = self.schema_li[idn1]
            if pred==label:
                self.right_count+=1
            else:
                self.error_count+=1









