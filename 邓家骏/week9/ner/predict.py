import torch
from model import TorchModel
from config import Config
from collections import defaultdict
from transformers import BertTokenizer
import re

def main(config):
    model = TorchModel(config)
    model.load_state_dict(torch.load(config['model_path']))
    while True:
        sentence = input("输入要预测的句子：")
        result = predict(model,config,sentence)
        print(result)
        


def predict(model,config,sentence):
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    inputs = tokenizer.encode(sentence,max_length=config["max_length"], truncation=True, padding='max_length')
    model.eval()
    output = []
    with torch.no_grad():
        output = model(torch.LongTensor(inputs).unsqueeze(0))[0]

    sentence = "$" + sentence
    output = "".join([str(x) for x in output[:len(sentence)]])
    print(output)
    results = defaultdict(list)
    for location in re.finditer("(04+)", output):
        s, e = location.span()
        results["LOCATION"].append(sentence[s:e])
    for location in re.finditer("(15+)", output):
        s, e = location.span()
        results["ORGANIZATION"].append(sentence[s:e])
    for location in re.finditer("(26+)", output):
        s, e = location.span()
        results["PERSON"].append(sentence[s:e])
    for location in re.finditer("(37+)", output):
        s, e = location.span()
        results["TIME"].append(sentence[s:e])
        
    return results


if __name__ == "__main__":
    Config["model_path"] = r"D:\code\week9\ner\model_output\num_layer_12-epoch_25-use_crf_True-learning_rate_1e-05.pth"
    main(Config)