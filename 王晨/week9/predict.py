import torch
from transformers import BertTokenizer, BertForTokenClassification
import os


def load_model(config):
    model = BertForTokenClassification.from_pretrained(config["bert_path"], num_labels=config["class_num"])
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % config["epoch"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def predict_and_save(input_texts, config, model, tokenizer):
    model.eval()
    predictions = []

    for text in input_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=config["max_length"])
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_labels = torch.argmax(logits, dim=-1).cpu().numpy().tolist()[0]
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu().numpy())
        predictions.append((tokens, predicted_labels))

    with open("predictions.txt", "w", encoding="utf-8") as f:
        for tokens, labels in predictions:
            for token, label in zip(tokens, labels):
                if token in ["[PAD]", "[CLS]", "[SEP]"]:
                    continue
                f.write(f"{token}\t{label}\n")
            f.write("\n")

    print("预测结果已保存到 predictions.txt 文件中")


# 使用示例
if __name__ == "__main__":
    from config import Config

    tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])
    model = load_model(Config)

    # 需要预测的文本
    input_texts = ["输入你的文本1", "输入你的文本2"]

    predict_and_save(input_texts, Config, model, tokenizer)
