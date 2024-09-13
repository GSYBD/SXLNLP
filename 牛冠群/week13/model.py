from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 加载预训练模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=9)

# 加载NER数据集
dataset = load_dataset("conll2003")

# 简单的数据预处理
def preprocess_data(examples):
    return tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)

# 预处理数据集
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Lora配置
lora_config = LoraConfig(r=4, lora_alpha=16)

# 应用Lora到模型
model = get_peft_model(model, lora_config)

from transformers import Trainer, TrainingArguments

# 训练参数设置
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=1,  # 只训练1个epoch
)

# Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# 训练模型
trainer.train()

trainer.evaluate(tokenized_dataset["test"])
