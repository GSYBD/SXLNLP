import torch.cuda
from loader import load_data
from config import Config
from model import SiameseNetwork, choose_optimizer

"""

    模型效果测试

"""


class Predictor:
    def __init__(self, config, model, knwb_data):
        self.config = config
        self.model = model
        self.train_data = knwb_data
        self.question_idx_to_standard_question_idx = {}
        self.question_ids = []
        self.vocab = self.train_data.dataset.vocab
        self.schema = self.train_data.dataset.schema
        self.idx_to_standard_question = dict((y, x) for x, y in self.schema.items())
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.knwb_to_vector()

    def knwb_to_vector(self):
        for standard_question_idx, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_idx_to_standard_question_idx[len(self.question_ids)] = standard_question_idx
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrices = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrices = question_matrices.cuda()
            self.knwb_vectors = self.model(question_matrices)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        return input_id

    def predict(self, sentence):
        input_id = self.encode_sentence(sentence)
        input_id = torch.LongTensor([input_id])
        if torch.cuda.is_available():
            input_id = input_id.cuda()
        with torch.no_grad():
            test_question_vector = self.model(input_id)
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))
            hit_index = self.question_idx_to_standard_question_idx[hit_index]
        return self.idx_to_standard_question[hit_index]


if __name__ == '__main__':
    knwb_data = load_data(Config['train_data_path'], Config)
    model = SiameseNetwork(Config)
    model.load_state_dict(torch.load('model_output/epoch_20.pth'))
    predictor = Predictor(Config, model, knwb_data)

    while True:
        # sentence = '固定宽带服务器密码修改'
        sentence = input('请输入问题: ')
        res = predictor.predict(sentence)
        print(res)
