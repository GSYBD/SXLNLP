# -*- coding: utf-8 -*-
import logging
import random

import torch
import numpy as np


from model import XModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XEvaluate():
    def __init__(self, config):
        self.config = config

    # 文本生成测试代码
    def generate_sentence(self,openings, model, tokenizer, window_size):
        # reverse_vocab = dict((y, x) for x, y in vocab.items())
        model.eval()
        with torch.no_grad():
            pred_char = ""
            pred_char_fll = ""
            # 生成了换行符，或生成文本超过30字则终止迭代
            while pred_char != "\n" and len(openings) <= window_size:
                openings += pred_char
                x = tokenizer.encode(openings, add_special_tokens=False)
                x = torch.LongTensor([x])
                if torch.cuda.is_available():
                    x = x.cuda()
                y = model(x)[0][-1]
                index = sampling_strategy(y)
                pred_char = ''.join(tokenizer.decode(index))
                pred_char_fll += pred_char
        return openings,pred_char_fll

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)