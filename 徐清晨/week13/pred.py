import torch
import logging
from model import TorchModel
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

from evaluate import Evaluator
from config import Config


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#大模型微调策略
# tuning_tactics = Config["tuning_tactics"]
tuning_tactics = "lora_tuning"

print("正在使用 %s"%tuning_tactics)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)

#重建模型
model = TorchModel(Config)
# print(model.state_dict().keys())
# print("====================")

model = get_peft_model(model, peft_config)
# print(model.state_dict().keys())
# print("====================")

state_dict = model.state_dict()

#将微调部分权重加载
loaded_weight = torch.load('model_output/lora.pth')

print(loaded_weight.keys())
state_dict.update(loaded_weight)

#权重更新后重新加载到模型
model.load_state_dict(state_dict)

#进行一次测试
# model = model.cuda()
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)