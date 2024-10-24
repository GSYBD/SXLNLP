
# -*- coding: utf-8 -*-
import re
import json
import pandas
import os

'''
基于脚本的多轮对话系统
1.nlu
2.dst
3.nlg
'''

class DialogSystem:
    def __init__(self):
        self.load()

    def load(self):
        # 加载场景
        self.node_id_to_node_info = {}
        self.load_scenario("scenario/scenario-买衣服.json")

        # 加载槽位模板
        self.slot_info = {}
        self.slot_template("scenario/slot_fitting_templet.xlsx")

    def load_scenario(self, scenario_file):
        scenario_name = os.path.basename(scenario_file).split('.')[0]
        print(scenario_name)
        with open(scenario_file, 'r', encoding='utf-8') as f:
            self.scenario = json.load(f)
        for node in self.scenario:
            node_id = node["id"]
            node_id = scenario_name + '-' + node_id
            if "childnode" in node:
                new_child  = []
                for child in node.get("childnode", []):
                    child = scenario_name + '-' + child
                    new_child.append(child)
                node["childnode"] = new_child
            self.node_id_to_node_info[node_id] = node
        print("场景加载完成")

    def slot_template(self, slot_template_file):
        df = pandas.read_excel(slot_template_file)
        for index, row in df.iterrows():
            slot = row["slot"]
            query = row["query"]
            value = row["values"]
            self.slot_info[slot] = [query, value]
        return

    def nlu(self, memory):
        '''
        1.意图识别
        2.语义槽填充
        '''
        memory = self.get_intent(memory)
        memory = self.get_slot(memory)
        return memory

    def get_intent(self, memory):
        '''
        意图识别
        根据用户输入，找到最匹配的意图   #从所有当前可以访问的节点中找到最高分节点
        '''
        user_input = memory.get("user_input")
        max_score = -1
        hit_intent = None
        for node_id in memory["available_nodes"]:
            score = self.cal_score(user_input, self.node_id_to_node_info[node_id]["intent"])
            if score > max_score:
                hit_intent = node_id
                max_score = score
        memory["hit_intent"] = hit_intent
        return memory

    def cal_score(self, user_input, intent):
        # 和单个节点计算得分
        scores = []
        for node_intent in intent:
            sentence_similarity = self.get_sentence_similarity(user_input, node_intent)
            scores.append(sentence_similarity)
        return max(scores)

    def get_sentence_similarity(self, sentence1, sentence2):
        # 计算两个句子的相似度
        # 这里可以使用一些文本相似度计算方法，比如余弦相似度、Jaccard相似度等
        # jaccard相似度计算
        set1 = set(sentence1)
        set2 = set(sentence2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)


    def get_slot(self, memory):
        #槽位抽取
        hit_intent = memory["hit_intent"]
        if hit_intent is not None:
           for slot in self.node_id_to_node_info[hit_intent].get("slot", []):
              _, values = self.slot_info[slot]
              if re.search(values, memory["user_input"]):
                memory[slot] = re.search(values, memory["user_input"]).group()
        return memory


    def dst(self, memory):
    # 对话状态跟踪，判断当前intent所需的槽位是否已经被填满
        hit_intent = memory["hit_intent"]
        slots = self.node_id_to_node_info[hit_intent].get("slot", [])
        for slot in slots:
            if slot not in memory:
                memory["need_slot"] = slot
                return memory
        memory["need_slot"] = None
        return memory

    def policy(self, memory):
        # 对话策略，判断当前对话状态，选择下一步的对话动作
        if "repeat" == memory["user_input_type"]:
            memory["action"] = "repeat"
            # 停留在当前节点
            memory["available_nodes"] = [memory["hit_intent"]]
        elif memory["need_slot"] is None:
            memory["action"] = "answer"
            memory["available_nodes"] = self.node_id_to_node_info[memory["hit_intent"]].get("childnode", [])
        else:
            memory["action"] = "request"
            # 停留在当前节点
            memory["available_nodes"] = [memory["hit_intent"]]
        return memory

    def nlg(self, memory):
        if memory["action"] == "answer":
            response = self.node_id_to_node_info[memory["hit_intent"]]["response"]
            answer = self.replace_slot(response, memory)
            memory["result"] = answer
        elif memory["action"] == "request":
            # 反问
            slot = memory["need_slot"]
            query, _ = self.slot_info[slot]
            memory["result"] = query
        elif memory["action"] == "repeat":
            # 重复上一次的回答
            memory["result"] = memory.get("last_response", "对不起，我没有记录上一次的回答，请稍后再试。")
        return memory["result"]

    def replace_slot(self, text, memory):
        hit_intent = memory["hit_intent"]
        slots = self.node_id_to_node_info[hit_intent].get("slot", [])
        for slot in slots:
            text = text.replace(slot, memory[slot])
        return text






    def generate_response(self, memory, user_input):
        '''
        1.意图识别
        2.语义槽填充
        3.对话状态跟踪
        4.对话策略
        5.文本生成
        '''
        memory["user_input"] = user_input
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.policy(memory)
        result = self.nlg(memory)
        memory["last_response"] = result  # 保存上一次的回答
        return result,memory


def get_sentence_similarity(user_input, sentenceList):
        # 用户输入与sentenceList匹配
        # 正则 模糊匹配
        for sentence in sentenceList:
            if sentence in user_input:
                return True
        return False


if __name__ == "__main__":
    sentenceList = ["再说一下", "再说一遍", "再说一次", "上次回答", "没听清", "没听懂", "重复一遍", "重复一次", "重复",
                    "再重复一遍", "再重复一次", "再重复", "再讲一遍", "再讲一次", "再讲", "再来一遍", "再来一次",
                    "再来", "再回答一遍", "再回答一次", "再回答", "再解释一遍", "再解释一次", "再解释", "再描述一遍",
                    "再描述一次", "再描述", "再介绍一遍", "再介绍一次", "再介绍", "再讲解一遍", "再讲解一次", "再讲解",
                    "再说明一遍", "再说明一次", "再说明", "再解释一遍", "再解释一次", "再解释", "再描述一遍",
                    "再描述一次", "再描述",
                    "再介绍一遍", "再介绍一次", "再介绍", "再讲解一遍", "再讲解一次", "再讲解", "再说明一遍",
                    "再说明一次", "再说明", "再解释一遍", "再解释一次", "再解释", "再描述一遍", "再描述一次", "再描述",
                    "再介绍一遍", "再介绍一次"]

    ds = DialogSystem()
    memory = {}
    memory["available_nodes"] = ["scenario-买衣服-node1"]
    while True:
       user_input = input("请输入：")
       print(get_sentence_similarity(user_input, sentenceList))
       if get_sentence_similarity(user_input, sentenceList):
            memory["user_input_type"] = "repeat"
       else:
            memory["user_input_type"] = "normal"
       # print(memory)
       result,memory = ds.generate_response(memory, user_input)
       print('bot:', result)
       print(memory)
