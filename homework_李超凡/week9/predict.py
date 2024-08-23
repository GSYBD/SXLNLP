import re
import torch
from loader import load_data
from config import Config
from model import TorchModel
from collections import defaultdict

class Predictor:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.vocab = load_vocab(config["vocab_path"])
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()


    def encode_sentence(self,sentence,padding=True):
        input_id = []
        for char in sentence:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return torch.LongTensor([input_id])

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def predict(self,sentence):
        input_ids=self.encode_sentence(sentence)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        with torch.no_grad():
            pred_results = self.model(input_ids)
            pred_results = torch.argmax(pred_results, dim=-1)
            pred_label = pred_results.cpu().detach().tolist()[0]
            pred_entities = self.decode(sentence, pred_label)
        return pred_entities

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict

if __name__ == "__main__":
    knwb_data = load_data(Config["train_data_path"], Config)
    model = TorchModel(Config)
    model.load_state_dict(torch.load("model_output/epoch_20.pth"))
    pd = Predictor(Config, model)

    while True:
        sentence = input("请输入语料：")
        res = pd.predict(sentence)
        print(res)
