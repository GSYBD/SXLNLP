from transformers import BertTokenizer
from model import TorchModel
from loader import load_vocab, load_schema
import torch
import re
from collections import defaultdict
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

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
        self.schema_list = ['LOCATION', 'ORGANIZATION', 'PERSON', 'TIME']

        # 加载PEFT训练的模型参数
        tuning_tactics = Config["tuning_tactics"]
        if tuning_tactics == "lora_tuning":
            peft_config = LoraConfig(
                r=8,
                lora_alpha=64,
                lora_dropout=0.1,
                target_modules=["query", "key", "value"]
            )
        elif tuning_tactics == "p_tuning":
            peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        elif tuning_tactics == "prompt_tuning":
            peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
        elif tuning_tactics == "prefix_tuning":
            peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

        self.model = TorchModel
        self.model = get_peft_model(self.model, peft_config)


        state_dict = self.model.state_dict()

        # 将微调部分权重加载
        if tuning_tactics == "lora_tuning":
            loaded_weight = torch.load(model_path)
        elif tuning_tactics == "p_tuning":
            loaded_weight = torch.load(model_path)
        elif tuning_tactics == "prompt_tuning":
            loaded_weight = torch.load(model_path)
        elif tuning_tactics == "prefix_tuning":
            loaded_weight = torch.load(model_path)
        state_dict.update(loaded_weight)
        self.model.load_state_dict(state_dict)
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
            input_id = self.tokenizer.encode(sentence, max_length=self.config['max_length'], pad_to_max_length=True)
        else:
            input_id = self.encode_sentence(sentence)

        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            pred = self.model(input_id)[0][0]
        pred = torch.argmax(pred, dim=-1)
        pred = pred.cpu().detach().tolist()
        pred = "".join([str(c) for c in pred[:100]])
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

    predict = Predict(Config, './model_output/lora_tuning.pth')
    res = predict.pred("中国政府打算和南亚政府")
    print(res)


