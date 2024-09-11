import torch
from create_datas import Create_Datas, load_datas
from config import config
import re
from collections import defaultdict


class Evaluate:
    def __init__(self, config, train_or_test_path="test_path"):
        self.config = config
        self.train_or_test_path = train_or_test_path
        self.schema_dic_t = defaultdict(list)
        self.schema_dic_p = defaultdict(list)
        self.stats_dict = {"LOCATION": defaultdict(int), "TIME": defaultdict(int), "PERSON": defaultdict(int), "ORGANIZATION": defaultdict(int)}

    def evaluate(self, model):
        self.stats_dict = {"LOCATION": defaultdict(int), "TIME": defaultdict(int), "PERSON": defaultdict(int), "ORGANIZATION": defaultdict(int)}
        self.schema_dic_t = defaultdict(list)
        self.schema_dic_p = defaultdict(list)
        model.eval()
        with torch.no_grad():
            test_datas = Create_Datas(config, self.train_or_test_path)
            sentences = test_datas.sentences
            self.x = torch.stack([i[0] for i in test_datas])
            true_y_li = torch.stack([i[1] for i in test_datas]).tolist()
            predict_y_li = model(self.x)
            for sentence, true_y, predict_y in zip(sentences, true_y_li, predict_y_li):
                self.decode(y=true_y, sentence=sentence, schema_dic="schema_dic_t")
                self.decode(y=predict_y, sentence=sentence, schema_dic="schema_dic_p")
            for key in ["LOCATION", "ORGANIZATION", "PERSON", "TIME"]:
                schema_dic_t_li = self.schema_dic_t[key]
                schema_dic_p_li = self.schema_dic_p[key]
                for schema_dic_t, schema_dic_p in zip(schema_dic_t_li, schema_dic_p_li):
                    self.stats_dict[key]["正确数量"] += len([i for i in [schema_dic_t] if i in [schema_dic_p] and i])
                    self.stats_dict[key]['"样本实体数"'] += len([i for i in [schema_dic_t] if i])
                    self.stats_dict[key]['"样本预测数"'] += len([i for i in [schema_dic_p] if i])
            print(self.stats_dict)

    def decode(self, y, sentence, schema_dic):
        str_y = "".join(str(i) for i in y)
        str_y = '$' +str_y
        schema_dic = getattr(self, schema_dic)

        matches_found = False
        for i in re.finditer("(04+)", str_y):
            s, e = i.span()
            schema_dic["LOCATION"].append(sentence[s:e])
            matches_found = True
        if not matches_found:
            schema_dic["LOCATION"].append([])

        matches_found = False
        for i in re.finditer("(15+)", str_y):
            s, e = i.span()
            schema_dic["ORGANIZATION"].append(sentence[s:e])
            matches_found = True
        if not matches_found:
            schema_dic["ORGANIZATION"].append([])

        matches_found = False
        for i in re.finditer("(26+)", str_y):
            s, e = i.span()
            schema_dic["PERSON"].append(sentence[s:e])
            matches_found = True
        if not matches_found:
            schema_dic["PERSON"].append([])

        matches_found = False
        for i in re.finditer("(37+)", str_y):
            s, e = i.span()
            schema_dic["TIME"].append(sentence[s:e])
            matches_found = True
        if not matches_found:
            schema_dic["TIME"].append([])

    """
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    """
