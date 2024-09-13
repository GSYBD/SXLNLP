import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch

class DataGenerator(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        self.title_list = []
        self.content_list = []
        self.attention_mask = []
        self.data_set = None
        self.data_loader = None
        self.load_data()

    def load_data(self):
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                dict_data = json.loads(line)
                title, content = dict_data.values()
                sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
                eos_token_id = self.tokenizer.convert_tokens_to_ids('[EOS]')
                title_token = self.tokenizer(title, padding='max_length', truncation=True, max_length=128, return_tensors='pt', add_special_tokens=False)
                content_token = self.tokenizer(content, padding='max_length', truncation=True, max_length=128, return_tensors='pt', add_special_tokens=False)
                input_ids = torch.cat([title_token['input_ids'], torch.tensor([[sep_token_id]]), content_token['input_ids']], dim=1)
                label_ids = torch.cat([torch.full((1, 128), -100), content_token['input_ids'], torch.tensor([[eos_token_id]])], dim=1)
                self.title_list.append(input_ids)
                self.content_list.append(label_ids)
            front_mask = torch.ones(257, 128, dtype=torch.long)
            back_mask = torch.cat([torch.zeros((128, 129), dtype=torch.long), torch.tril(torch.ones((129, 129), dtype=torch.long))], dim=0)
            self.attention_mask = torch.cat([front_mask, back_mask], dim=1).expand(20, -1, -1)
            # self.data_set = Dataset(self.title_list, self.content_list)
            # self.data_loader = DataLoader(self.data_set, batch_size=20, shuffle=True)

    def __len__(self):
        return len(self.title_list)
    

    def __getitem__(self, idx):
        return [self.title_list[idx], self.content_list[idx]]
    
def get_loader(file_path):
    data_generator = DataGenerator(file_path)
    return DataLoader(data_generator, batch_size=10, shuffle=True), data_generator
