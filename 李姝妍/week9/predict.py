from transformers import BertTokenizer
from model import TorchModel
from loader import load_vocab, load_schema
import torch
import re
from collections import defaultdict
class Predict:
    def __init__(self, config, model_path):
        self.config = config
        self.use_bert = config['use_bert']
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
        self.vocab = load_vocab(config['vocab_path'])
        config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])
        config['class_num'] = len(self.schema)
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.schema_list = ['LOCATION', 'ORGANIZATION', 'PERSON', 'TIME']
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    def encode_sentence(self, text):
        input_id = [self.vocab.get(c, self.vocab['[UNK]']) for c in text]
        return self.padding(input_id)

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config['max_length']]
        input_id += [pad_token] * (self.config['max_length'] - len(input_id))
        return input_id

    def pred(self, sentence):
        if self.use_bert:
            input_id = self.tokenizer.encode(sentence, add_special_tokens=False, max_length=self.config['max_length'],
                                             padding='max_length',
                                             truncation=True)
        else:
            input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            pred = self.model(input_id)[0]
        pred = torch.argmax(pred, dim=-1)
        pred = pred.detach().cpu().tolist()
        pred = "".join([str(c) for c in pred[:len(sentence)]])
        res = defaultdict(list)
        for locations in re.finditer("(04+)", pred):
            s, e = locations.span()
            res['LOCATION'].append(sentence[s: e])
        for locations in re.finditer("(15+)", pred):
            s, e = locations.span()
            res['ORGANIZATION'].append(sentence[s: e])
        for locations in re.finditer("(26+)", pred):
            s, e = locations.span()
            res['PERSON'].append(sentence[s: e])
        for locations in re.finditer("(37+)", pred):
            s, e = locations.span()
            res['TIME'].append(sentence[s: e])
        return res



if __name__ == '__main__':
    from config import Config
    predict = Predict(Config, './model_output/epoch_30.pth')
    res = predict.pred("国务院副总理李强在十月三日在人民大会堂主持召开了第十次人民代表大会,并明确指出在明年上半年实现全面脱贫")
    print(res)
