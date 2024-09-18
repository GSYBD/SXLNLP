import os
import torch
import numpy as np
from loader import load_data
from model import TorchModel
from config import Config
from evaluate import Evaluator
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
from model import choose_optimizer

def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = TorchModel(config)

    # 根据不同的 tuning_tactics 应用不同的微调方法
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

    # 使用PEFT方法微调模型
    model = get_peft_model(model, peft_config)

    if tuning_tactics == "lora_tuning":
        # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
        # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        for param in model.get_submodule("model").get_submodule("classifier").parameters():
            param.requires_grad = True

    # 判断是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model)

    # 开始训练
    for epoch in range(config["epoch"]):
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                print(f"Batch {index}, Loss: {loss.item()}")

        print(f"Epoch {epoch + 1} average loss: {np.mean(train_loss)}")
        evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], f"epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    config = Config()
    main(config)
