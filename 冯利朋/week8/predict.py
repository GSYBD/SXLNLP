from transformers import BertTokenizer
from model import SeeNetWork
from loader import load_vocab, load_schema, load_data
import torch
class Predict:
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path
        self.train_data = load_data(config['train_data_path'], config)
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])
        self.index_to_schema = {y:x for x, y in self.schema.items()}
        self.use_bert = config['use_bert']
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.model = SeeNetWork(config)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.load_know()
    def load_know(self):
        self.question_index_to_stander_index = {}
        self.question_ids = []
        for target, questions in self.train_data.dataset.know.items():
            for question in questions:
                self.question_index_to_stander_index[len(self.question_ids)] = target
                self.question_ids.append(question)
        self.question_ids = torch.stack(self.question_ids, dim=0)
        with torch.no_grad():
            if torch.cuda.is_available():
                self.question_ids = self.question_ids.cuda()
            self.know_vectors = self.model(self.question_ids)
            self.know_vectors = torch.nn.functional.normalize(self.know_vectors, dim=-1)

    def encode_sentence(self, text):
        input_id = [self.vocab.get(c, self.vocab['[UNK]']) for c in text]
        return self.padding(input_id)

    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

    def pre(self, sentence):
        if self.use_bert:
            input_id = self.tokenizer.encode(sentence, max_length=self.config['max_length'], pad_to_max_length=True)
        else:
            input_id = self.encode_sentence(sentence)
        with torch.no_grad():
            input_id = torch.LongTensor([input_id])
            pred_res = self.model(input_id)
            res = torch.mm(pred_res.unsqueeze(0), self.know_vectors.T)
            hit_index = int(torch.argmax(res))
            hit_index = self.question_index_to_stander_index[hit_index]
            print(self.index_to_schema[hit_index])
        return
