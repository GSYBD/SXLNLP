import torch
from transformers import BertTokenizer
class EncodeUtil:
    def __init__(self, use_bert, vocab, max_length=None, bert_path=None):
        self.use_bert = use_bert
        self.vocab = vocab
        self.max_length = max_length
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def encode_sentence(self, text, padding=True):
        if self.use_bert:
            inputs = self.tokenizer(text, add_special_tokens=False, max_length=self.max_length, truncation=True,
                                  padding='max_length')
            return inputs['input_ids'], inputs['attention_mask']
            # return self.tokenizer.encode(text, add_special_tokens=False, max_length=self.max_length, truncation=True, padding='max_length')
        input_id = [self.vocab.get(c, self.vocab['[UNK]']) for c in text]
        if padding:
            input_id = self.padding(input_id)
        return input_id, torch.zeros(self.max_length)

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.max_length]
        input_id += [pad_token] * (self.max_length - len(input_id))
        return input_id







