import torch
from transformers import BertModel, BertConfig

config_path = './bert-base-chinese/config.json'
config = BertConfig.from_json_file(config_path)
model = BertModel(config)
model_path = './bert-base-chinese/pytorch_model.bin'
model.load_state_dict(torch.load(model_path), strict=False)
model.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of trainable parameters: {total_params}")
# Total number of trainable parameters: 24301056

