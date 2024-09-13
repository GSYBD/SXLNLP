import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

train_data = load_data(Config["train_data_path"], Config)

print(train_data)
      
for index, batch_data in enumerate(train_data):
    print(index, batch_data)
    input_id1, input_id2, labels = batch_data
    print(input_id1,"\n", input_id2, "\n", labels)
    print(type(input_id1),"\n", input_id2, "\n", labels)