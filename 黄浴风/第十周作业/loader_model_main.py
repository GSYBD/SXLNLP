#loader.py
# loader.py
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.data = self.load_data(data_path)
    
    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        input_ids = []
        for line in lines:
            line = line.strip()
            if line:
                encoded = self.tokenizer.encode_plus(
                    line,
                    max_length=self.config["max_length"],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids.append(encoded['input_ids'].squeeze(0))
        return input_ids

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

def load_data(data_path, config, shuffle=True):
    dataset = DataGenerator(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader


#model.py

# model.py
import torch
import torch.nn as nn
from transformers import BertModel

class BertAutoregressiveModel(nn.Module):
    def __init__(self, config):
        super(BertAutoregressiveModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(config["hidden_size"], config["class_num"])

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.bert.config.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss
        else:
            return logits

#main.py
# main.py
import torch
import logging
from config import Config
from model import BertAutoregressiveModel
from loader import load_data
from transformers import AdamW

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_epoch(model, train_data, optimizer, config):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_data):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = (input_ids != model.bert.config.pad_token_id).float()
        
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if step % 100 == 0 and step != 0:
            logger.info(f"Step {step}, Loss: {loss.item()}")
    return total_loss / len(train_data)

def main(config):
    train_data = load_data(config["train_data_path"], config)
    model = BertAutoregressiveModel(config)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(config["epoch"]):
        avg_loss = train_epoch(model, train_data, optimizer, config)
        logger.info(f"Epoch {epoch + 1}/{config['epoch']}, Average Loss: {avg_loss}")
        
        # 保存模型
        torch.save(model.state_dict(), f"{config['model_path']}/model_epoch_{epoch + 1}.bin")

if __name__ == "__main__":
    main(Config)
