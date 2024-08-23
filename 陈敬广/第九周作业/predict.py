import torch

from model import NerModel
from loader import text_to_seq, load_vocab, load_schema

'''
模型训练完成，模型预测

'''


class Predict:
    def __init__(self, config, model_path):
        # 模型配置
        self.config = config

        # 加载相同的词表
        self.tokenizer = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.tokenizer.vocab)
        self.schema = load_schema(self.config['schema_path'])
        self.config["class_num"] = len(self.schema)
        self.index_to_label = dict((y, x) for x, y in self.schema.items())

        # 模型参数路径
        self.model_path = model_path
        # 加载模型
        self.model = NerModel(config)
        self.model.load_state_dict(torch.load(self.model_path))



    def predict(self, text):
        # 输出文本
        out_text = []
        # 将输入的文本转成序列
        input_seq = text_to_seq(self.tokenizer,text,self.config['max_length'])
        input_seq = torch.LongTensor([input_seq])
        with torch.no_grad():
            y_pred = self.model.forward(input_seq)  # shape (1,text_len,class_num)
            y_pred = torch.argmax(y_pred, dim=-1).squeeze()  # shape(text_len,)
            y_pred = y_pred.tolist()

        for index, char in enumerate(text):
            char += self.index_to_label[y_pred[index+1]]
            out_text.append(char)
        out_text = ''.join(out_text)
        return out_text


if __name__ == '__main__':
    from config import Config
    p = Predict(Config,'model_output/model.pth')
    out_text = p.predict('陈墨周六去北京')
    print(out_text)
