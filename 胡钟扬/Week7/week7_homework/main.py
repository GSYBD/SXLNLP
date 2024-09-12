import torch
import os
import random
import numpy as np
import logging


from model import TorchModel, choose_optimizer
from loader import load_data
from evaluate import Evaluator
from config import Config
from typing import Dict, List, Tuple
from collections import defaultdict
from transformers import BertModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def main(config):
    
    # 创建保存模型权重的目录
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    
    
    # load_data必须在创建模型之前，否则有些字段读不到
    train_data = load_data(config['train_data_path'],config)
        
    
    model = TorchModel(config)
    
    
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)

    cuda_flag = torch.cuda.is_available()
    
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()


    epoch_num = config['epoch']
    
    
    
    for epoch in range(epoch_num):
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch in enumerate(train_data):
            
            if cuda_flag:
               batch = [d.cuda() for d in batch]
            
            # 分离特征和label
            input_ids, labels = batch
            optimizer.zero_grad()
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)

        logger.info("epoch %d loss %f" % (epoch, np.mean(train_loss, axis=0)))
        acc = evaluator.eval(epoch)
    
    
    model_path = os.path.join(config["model_path"], config["model_type"]+ ".pth" )
    torch.save(model.state_dict(), model_path)  #保存模型权重
    
    return acc, evaluator.stats_dict['prediction_time'],evaluator.stats_dict['avg_example_len']
            
    
    
    
    
def compare_models_by_train(models:List[str], config):
    
    model_stats=[]
    for model in models:
        config["model_type"] = model
        accuracy, pred_time, avg_example_len = main(config)
        # print("最后一轮准确率：", main(config), "当前配置：", config["model_type"])
        report = "最后一轮准确率：%.2f, 当前配置：%s" % (accuracy, config["model_type"])
        model_stats.append({"model_type":model, "acc":accuracy, "time":pred_time, "sample_len":avg_example_len}) 
        print(report)
    
    # 将所有模型的统计信息写入文件
    from csv_to_json import write_dicts_to_csv
    write_dicts_to_csv(model_stats, "week7_homework/model_output.csv")
    
    

# def compare_models_by_evaluate(models:List[str], config):
#     model_stats=[]
    
#     # 加载权重
#     for file_name in os.listdir(config["model_path"]):
#         file_path = os.path.join(config["model_path"], file_name)   
        
#         # 检查是否为文件且以 .pt 或 .pth 结尾
#         if os.path.isfile(file_path) and (file_name.endswith("pt") or file_name.endswith("pth")):
#             print(f"Found model file: {file_path}") 
#             print("model name = ", file_name.split('.')[0])
#             config['model_type'] = file_name.split('.')[0]
#             # load_data必须在创建模型之前，否则有些字段读不到
#             val_data = load_data(config['valid_data_path'],config)
#             model = TorchModel(config)
            
#             states = torch.load(file_path)
#             for key in states.keys():
#                 print(key)
            
#             # 如何是bert类型的
#             if model.use_bert:
#                 model.encoder.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
#             # 遍历对象的所有成员  
#             for layer_name in dir(model.encoder):  
#                 # 排除内置的特殊方法和属性  
#                 if not layer_name.startswith("bert"): 
#                     layer = getattr(model.encoder, layer_name)
#                     layer.load_dict(states)
#                     # setattr(model.encoder, layer_name, layer.load_dict(states)) 
                    
#             # model.encoder.load_state_dict(states)
            
#             evaluator = Evaluator(config, model, logger)
#             acc = evaluator.eval(config['epoch'])
#             report = "最后一轮准确率：%.2f, 当前配置：%s" % (acc, config["model_type"])
#             model_stats.append({"model_type":config['model_type'], "acc":acc, "time": evaluator.stats_dict['prediction_time'], "sample_len":evaluator.stats_dict['avg_example_len']}) 
#             print(report)
            
#     # 将所有模型的统计信息写入文件
#     from csv_to_json import write_dicts_to_csv
#     write_dicts_to_csv(model_stats, "week7_homework/model_output_eval.csv")

def compare_models_by_hyperparameters():
        #对比所有模型
        #中间日志可以关掉，避免输出过多信息
        # 超参数的网格搜索
        for model in ["gated_cnn"]:
            Config["model_type"] = model
            for lr in [1e-3, 1e-4]:
                Config["learning_rate"] = lr
                for hidden_size in [128]:
                    Config["hidden_size"] = hidden_size
                    for batch_size in [64, 128]:
                        Config["batch_size"] = batch_size
                        for pooling_style in ["avg"]:
                            Config["pooling_style"] = pooling_style
                            print("最后一轮准确率：", main(Config), "当前配置：", Config)


if __name__ == "__main__":

    # main(Config)
    
    
    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])
    
    models = Config["model_list"]
    # models = ['bert_mid_layer']
    compare_models_by_train(models, Config)
    # compare_models_by_evaluate(models, Config)

