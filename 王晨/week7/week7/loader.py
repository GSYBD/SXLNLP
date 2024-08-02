import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch

def load_data(file_path, max_len, test_size, random_state):
    data = pd.read_csv(file_path)
    data.columns = ['label', 'review']
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    def tokenize_texts(texts, max_len):
        texts = texts.tolist()
        encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        return encodings
    
    encodings = tokenize_texts(data['review'], max_len)
    labels = torch.tensor(data['label'].values)
    x_train, x_test, y_train, y_test = train_test_split(encodings['input_ids'], labels, test_size=test_size, random_state=random_state)
    
    return x_train, x_test, y_train, y_test
