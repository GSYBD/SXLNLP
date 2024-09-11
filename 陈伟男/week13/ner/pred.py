import torch
import logging
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from config import Config
from peft import get_peft_model, LoraConfig
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
    #加载模型
    model = TorchModel(config)
    model = get_peft_model(model, peft_config)
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    state_dict = model.state_dict()
    loaded_weight = torch.load('output/lora_tuning.pth')
    state_dict.update(loaded_weight)
    #权重更新后重新加载到模型
    model.load_state_dict(state_dict)
    evaluator.eval()
if __name__ == '__main__':
    main(Config)