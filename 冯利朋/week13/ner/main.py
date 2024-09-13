import os.path
import random
import torch.nn as nn
import torch.cuda
import numpy as np
from loader import load_data
from model import TorchModel, choose_optimizer
from evaluate import Evaluate
import logging
from config import Config
from peft import LoraConfig, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, get_peft_model
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def main(config):
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    epoch_num = config['epoch']
    train_data = load_data(config['train_data_path'], config)
    model = TorchModel

    # 微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == 'lora_tuning':
        peft_config = LoraConfig(
            r=8,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

    model = get_peft_model(model, peft_config)
    # 那些层不用冻结
    if tuning_tactics == "lora_tuning":
        for param in model.get_submodule('model').get_submodule('classifier').parameters():
            param.requires_grad = True

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = choose_optimizer(config, model)
    evaluate = Evaluate(config, model,logger)
    # 损失函数
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(1, epoch_num + 1):
        watch_loss = []
        model.eval()
        logger.info("开始训练第%d轮" % epoch)
        for index, batch_data in enumerate(train_data):
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            input_ids, labels = batch_data
            output = model(input_ids)[0]
            loss = loss_func(output.view(-1, output.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        logger.info("avg-loss=%f" % np.mean(watch_loss))
        acc = evaluate.eval(epoch)
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    # torch.save(model.state_dict(), model_path)
    save_tunable_parameters(model, model_path)
    return model, train_data
def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)

if __name__ == '__main__':
    main(Config)



 # import re
 #    pattern = r'\((\w+)\): Linear'
 #    linear_layers = re.findall(pattern, str(model.modules))
 #    target_modules = list(set(linear_layers))
 #    print(model.state_dict().keys())

