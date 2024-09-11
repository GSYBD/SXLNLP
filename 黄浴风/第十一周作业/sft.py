import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json

# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = load_data(r'E:\train.json')

# 将数据转换为Hugging Face Dataset格式
train_data = Dataset.from_dict({
    'input': [item['title'] for item in data],
    'output': [item['content'] for item in data]
})
# 加载ChatGLM模型和分词器
model_name = "THUDM/chatglm-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
def preprocess_function(examples):
    inputs = examples['input']
    outputs = examples['output']
    
    # 将输入和输出拼接在一起，形成模型的输入
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, max_length=512, truncation=True)

    # 将标签与输入分开
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# 预处理数据
train_data = train_data.map(preprocess_function, batched=True)
# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    predict_with_generate=True
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()
# 定义推理函数
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 示例生成
input_text = "输入一个新闻标题"
generated_content = generate_response(input_text)
print(generated_content)

