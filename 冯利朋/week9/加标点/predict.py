from model import TorchModel
from loader import load_vocab, load_schema
from encode_util import EncodeUtil
import torch
class Predict:
    def __init__(self, config, model_path):
        self.config = config
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])
        self.index_to_schema = {y: x for x, y in self.schema.items()}
        self.config['class_num'] = len(self.schema)
        self.use_bert = config['use_bert']
        self.encodeUtil = EncodeUtil(self.use_bert, self.vocab, config['max_length'], bert_path=config['pretrain_model_path'])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    def pre(self, sentence):
        input_id = self.encodeUtil.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_id)[0]
        pred = torch.argmax(pred, dim=-1)
        pred_sentence = ""
        for y, char in zip(pred, sentence):
            pred_sentence += char + self.index_to_schema[int(y)]
        return pred_sentence
if __name__ == '__main__':
    from config import Config
    predict = Predict(Config, './model_output/epoch_20.pth')
    res = predict.pre("来源:新周刊 向宫崎骏学习,绝不是把他的画风“汉化”过来。我们与日美")
    print(res)