import torch
import re
import csv
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from loader import load_data



class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data: DataLoader = load_data(config["valid_data_path"], config, shuffle=False)

    
    def eval(self, epoch):
        
        if type(self.model).__name__ == "WholeSentenceNERModel":
            self.eval_sentence_ner(epoch)
            return
        
        elif type(self.model).__name__ == "RegularExpressionModel":
            self.eval_regular_expression(epoch)
            return 
        
        self.logger.info(f"开始测试第{epoch}轮模型效果")
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON":defaultdict(int),
            "ORGANIZATION":defaultdict(int),
        }
        
        self.model.eval()
        
        for index, batch in enumerate(self.valid_data):
            # get a batch of sentences
            sentences = self.valid_data.dataset.sentences[index*self.config['batch_size']:(index+1)*self.config['batch_size']]
                
            if torch.cuda.is_available():
                batch = [d.cuda() for d in batch]
            
            input_ids_list, labels_list = batch # 1 tokens -> 1 label
            # input_ids_list: [batch_size, max_length]
            # labels_list: [batch_size, max_length]
            with torch.no_grad():
                # predict
                predicts = self.model(input_ids_list)
            self.write_stats(labels_list, predicts, sentences)
        self.show_stats()
        return
    
    
    def eval_sentence_ner(self,epoch):
        return
            
            
    
    def eval_regular_expression(self, epoch):
        return
    
    
    
    def write_stats(self, labels, predicts, sentences):
        '''
            处理一个 batch的数据，并写入统计字典，统计预测结果与真实结果的差异
            
            labels: [batch_size, max_length]
            predicts: [batch_size, max_length]
            sentences: [batch_size]
        '''
        assert len(labels)==len(predicts)==len(sentences)
        
        '''
            .cpu().detach().tolist() 的含义:
                .cpu()：将数据从 GPU 转移回 CPU，以便脱离计算图后再处理。
                .detach()：阻止梯度传播，避免影响后续的反向传播。
                .tolist()：将张量转换为普通的 Python 列表。
        '''
        if not self.config['use_crf']:
            predicts = torch.argmax(predicts, dim = -1)
        
        for true_label, pred_label, sentence in zip(labels, predicts, sentences):
            if not self.config['use_crf']:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            
            true_entities = self.decode(sentence, true_label)
            
            pred_entities =self.decode(sentence, pred_label)
            
            print("=============")
            print("true entities:\n",true_entities)
            
            print("============")
            print("pred entities:\n",pred_entities)
            
            # 记录当前sentence的各种统计参数

            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]['correct_pred_entity'] +=  len([entity for entity in pred_entities[key] if entity in true_entities[key]])   # TP
                self.stats_dict[key]['real_entity_num'] += len([entity for entity in true_entities[key]])   # TP + FN
                self.stats_dict[key]['total_pred_entity'] += len([entity for entity in pred_entities[key]]) # TP + FP
            
            
            return
            
    
    
    def show_stats(self):
        '''
            展示所有句子一起的实体识别正确率
        '''
        self.final_stats = {}
        self.final_stats['model_name'] = type(self.model).__name__
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]['correct_pred_entity']/(self.stats_dict[key]['total_pred_entity']+1e-5)
            recall = self.stats_dict[key]['correct_pred_entity'] / (self.stats_dict[key]['real_entity_num'] + 1e-5)
            F1= 2 * precision * recall / (precision + recall+1e-5)
            F1_scores.append(F1)
            self.final_stats[key] = {}
            self.final_stats[key]['precision'] = precision
            self.final_stats[key]['recall'] = recall
            self.final_stats[key]['F1'] = F1
            
            self.logger.info(f"Entity Type:{key}, precision:{precision:.4f}, recall:{recall:.4f}, F1:{F1:.4f}")
        
        macro_f1 = np.mean(F1_scores)
        self.logger.info(f"Macro-F1:{macro_f1:.4f}")
    
        correct_pred = sum([self.stats_dict[key]['correct_pred_entity'] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]['total_pred_entity'] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        real_entity = sum([self.stats_dict[key]['real_entity_num'] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall  = correct_pred / (real_entity + 1e-5)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall+1e-5)
        
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        
        
        # encapsulate every statistic into a dict
        final_stats_general = {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "avg_precision": np.mean([self.final_stats[key]['precision'] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]]),
            "avg_recall": np.mean([self.final_stats[key]['recall'] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]]),
        }
        
        self.final_stats.update(final_stats_general)
        
        
        self.write_stats_to_csv(self.final_stats)
        
        return
    def decode(self, sentence, labels):
        '''
            labels: seq_len * 1
            
            return dict({
                "LOCATION":['xxx','yyyy','zzzz'],
                "ORGNIZATION":[]
            })
        '''  
        
        # 确保标签的长度与句子一致，避免多余的标签。
        labels = "".join([str(label) for label in labels[:len(sentence)]])
        
        results = defaultdict(list)
        
        # (04+)：匹配一个 0 后跟一个或多个 4 的序列 [B-LOCATION + 多个 I-LOCATION]
        for location in re.finditer("(04+)", labels):
            s,e = location.span()
            results["LOCATION"].append(sentence[s:e])
        
        
        
        for orgnization in re.finditer("(15+)", labels):
            s,e = orgnization.span()
            results["ORGANIZATION"].append(sentence[s:e])
            
        for person in re.finditer("(26+)", labels):
            s,e = person.span()
            results["PERSON"].append(sentence[s:e])
        
        
        for time in re.finditer("(37+)",labels):
            s,e = time.span()
            results["TIME"].append(sentence[s:e])
        
        return results
    def write_stats_to_csv(self, stats_dict:dict):
        '''
         stats_dict:{
             "PERSON":{
                 "precision": precision,
                 "recall": recall,
                 "F1": F1
            }, 
             "LOCATION":{
                 ...
            }, 
             "TIME":{
                 ...
            }, 
             "ORGANIZATION":{
               ...
                 
            },
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall
         }
         
        '''
        with open('metrics.csv', 'a+', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.final_stats['model_name'], '', '', ''])
            writer.writerow(['Entity type', 'Precision', 'Recall', 'F1'])
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                precision = self.final_stats[key]["precision"] 
                recall = self.final_stats[key]["recall"]
                F1 = self.final_stats[key]["F1"]
                writer.writerow([key, f'{precision:.6f}', f'{recall:.6f}', f'{F1:.6f}'])
            writer.writerow(['Macro-F1', '', '', f'{self.final_stats["macro_f1"]:.6f}'])
            writer.writerow(['Micro-F1', '', '', f'{self.final_stats["micro_f1"]:.6f}'])
            writer.writerow(['Average Precision', '', '', f'{self.final_stats["avg_precision"]:.6f}'])
            writer.writerow(['Average Recall', '', '', f'{self.final_stats["avg_recall"]:.6f}'])
            writer.writerow([''])
            writer.writerow([''])
            
        print("statistics has been written to csv ~~~")
        
        return