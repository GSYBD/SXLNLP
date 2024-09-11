import json
import torch

def load_data(data_path, tokenizer, config, logger):
    # 加载并解析数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 确保 JSON 文件是一个数组

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for example in data:
        encoding = tokenizer.encode_plus(
            example['content'],  # 假设数据中有'content'字段
            max_length=config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids_list.append(encoding["input_ids"])
        attention_mask_list.append(encoding["attention_mask"])
        labels_list.append(torch.tensor(example.get('label', 0)))  # 如果没有 'label' 字段，可以给个默认值

    # 将列表转换为张量，并添加 batch 维度
    input_ids = torch.cat(input_ids_list, dim=0)  # (batch_size, sequence_length)
    attention_mask = torch.cat(attention_mask_list, dim=0)  # (batch_size, sequence_length)
    labels = torch.stack(labels_list, dim=0)  # (batch_size)

    return input_ids, attention_mask, labels