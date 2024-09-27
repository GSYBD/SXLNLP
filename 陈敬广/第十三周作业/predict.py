import torch

from model import NerModel
from loader import text_to_seq, load_vocab, load_schema
from peft import LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

'''
模型训练完成，模型预测

'''


class Predict:
    def __init__(self, config):
        # 模型配置
        self.config = config

        # 加载相同的词表
        self.tokenizer = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.tokenizer.vocab)
        self.schema = load_schema(self.config['schema_path'])
        self.config["class_num"] = len(self.schema)
        self.index_to_label = dict((y, x) for x, y in self.schema.items())
        self.peft_config = peft_config(config["tuning_tactics"])
        # 加载模型
        self.model = NerModel(config)
        self.load_weights(config["tuning_tactics"])



    def predict(self, text):
        self.model.eval()
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


    def load_weights(self,tuning_tactics):
        state_dict = self.model.state_dict()
        # 将微调部分权重加载
        loaded_weight = torch.load('model_output/lora_tuning.pth')
        if tuning_tactics == "p_tuning":
            loaded_weight = torch.load('model_output/p_tuning.pth')
        elif tuning_tactics == "prompt_tuning":
            loaded_weight = torch.load('model_output/prompt_tuning.pth')
        elif tuning_tactics == "prefix_tuning":
            loaded_weight = torch.load('model_output/prefix_tuning.pth')
        state_dict.update(loaded_weight)
        # 权重更新后重新加载到模型
        self.model.load_state_dict(state_dict)



def peft_config(tuning_tactics):

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
    if tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    return peft_config

if __name__ == '__main__':
    from config import Config
    p = Predict(Config)
    out_text = p.predict('陈墨周六去北京')
    print(out_text)
