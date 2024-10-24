import re
import pandas
import json
import os

'''
基于脚本的多轮对话
'''

class DialogSystem:
    def __init__(self):
        self.load()

    def load(self):
        #加载场景
        self.node_id_to_node_info = {}
        self.load_scenario('E:\AI\课程资料\第十六周 对话系统\week16 对话系统\scenario\scenario-买衣服.json')
        #加载槽位模板
        self.slot_info = {}
        self.slot_template('E:\AI\课程资料\第十六周 对话系统\week16 对话系统\scenario\slot_fitting_templet.xlsx')

    def load_scenario(self, scenario_path):
        scenario_name = os.path.basename(scenario_path).split('.')[0]
        with open(scenario_path, 'r', encoding='utf-8') as f:
            self.scenario = json.load(f)
        for node in self.scenario:
            node_id = node['id']
            node_id = scenario_name + '-' + node_id
            if 'childnode' in node:
                new_child = []
                for child in node['childnode']:
                    child = scenario_name + '-' + child
                    new_child.append(child)
            self.node_id_to_node_info[node_id] = node
            self.node_id_to_node_info[node_id]['childnode'] = new_child
        print('场景加载完成')

    def slot_template(self, template_path):
        df = pandas.read_excel(template_path)
        for index, row in df.iterrows():
            slot = row['slot']
            query = row['query']
            value = row['values']
            self.slot_info[slot] = [query, value]
        return

    def nlu(self, memory):
        #意图识别
        if self.is_repeat_request(memory['user_input']):
            memory['hit_intent'] = 'repeat'
        else:
            memory = self.get_intent(memory)
            #槽位抽取
            memory = self.get_slot(memory)
        return memory

    def is_repeat_request(self, user_input):
        # 简单判断用户是否要求重复问题
        repeat_phrases = ['再说一遍', '没听清', '能重复一下吗', '再重复一遍']
        for phrase in repeat_phrases:
            if phrase in user_input:
                return True
        return False

    def get_intent(self, memory):
        #从所有当前可以访问的节点中找到最高分的节点
        max_score = -1
        hit_intent = None
        for node_id in memory['available_nodes']:
            node = self.node_id_to_node_info[node_id]
            score = self.get_node_score(node, memory)
            if score > max_score:
                max_score = score
                hit_intent = node_id
        memory['hit_intent'] = hit_intent
        memory['hit_intent_score'] = max_score
        return memory

    def get_node_score(self, node, memory):
        #和单个节点计算得分
        intent = memory['user_input']
        node_intents = node['intent']
        scores = []
        for node_intent in node_intents:
            sentence_similarity = self.get_sentence_similarity(intent, node_intent)
            scores.append(sentence_similarity)
        return max(scores)

    def get_sentence_similarity(self, intent, node_intent):
        #计算句子相似度
        #jaccard相似度计算
        set1 = set(intent)
        set2 = set(node_intent)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def get_slot(self, memory):
        #槽位抽取
        hit_intent = memory['hit_intent']
        for slot in self.node_id_to_node_info[hit_intent].get('slot', []):
            _, values = self.slot_info[slot]
            print(values)
            if re.search(values, memory['user_input']):
                memory[slot] = re.search(values, memory['user_input']).group()
        return memory

    def dst(self, memory):
        #对话状态跟踪，判断当前intent所需的槽位是否已经被填满
        hit_intent = memory['hit_intent']
        if hit_intent == 'repeat':
            memory['need_slot'] = None
            return memory
        slots = self.node_id_to_node_info[hit_intent].get('slot', [])
        for slot in slots:
            if slot not in memory:
                memory['need_slot'] = slot
                return memory
        memory['need_slot'] = None
        return memory

    def policy(self, memory):
        if memory.get('repeat') == 'repeat':
            memory['action'] = 'repeat'
            return memory
        #对话策略，根据当前状态选择下一步动作
        #如果槽位有欠缺，反问槽位
        #如果槽位没有欠缺，直接回答
        if memory['hit_intent'] == 'repeat':
            memory['action'] = 'repeat'
        elif memory['need_slot'] is None:
            memory['action'] = 'answer'
            #开放子节点
            memory['available_nodes'] = self.node_id_to_node_info[memory['hit_intent']].get('childnode', [])
            #执行动作
            # self.take_action(memory)
        else:
            memory['action'] = 'ask'
            #停留在当前节点
            memory['available_nodes'] = [memory['hit_intent']]
        return memory


    def nlg(self, memory):
        #文本生成模块
        if memory['hit_intent'] == 'repeat':
            memory['bot_response'] = memory['last_bot_response']
        elif memory['action'] == 'answer':
            #直接回答
            answer = self.node_id_to_node_info[memory['hit_intent']]['response']
            memory['bot_response'] = self.replace_slot(answer, memory)
        else:
            #反问槽位
            slot = memory['need_slot']
            query, _ = self.slot_info[slot]
            memory['bot_response'] = query
        return memory

    def replace_slot(self, answer, memory):
        hit_intent = memory['hit_intent']
        slots = self.node_id_to_node_info[hit_intent].get('slot', [])
        for slot in slots:
            answer = answer.replace(slot, memory[slot])
        return answer

    def generate_response(self, user_input, memory):
        memory['user_input'] = user_input
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.policy(memory)
        memory = self.nlg(memory)
        memory['last_bot_response'] = memory['bot_response']  # 保存上次的响应
        return memory['bot_response']

if __name__ ==  '__main__':
    memory = {'available_nodes':['scenario-买衣服-node1']}
    ds = DialogSystem()
    print(ds.node_id_to_node_info)
    print(ds.slot_info )
    while True:
        user_input = input('用户的输入是：')
        response = ds.generate_response(user_input, memory)
        print('bot:', response)
