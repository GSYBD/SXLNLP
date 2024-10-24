import re
import os
import json
import pandas
class DialogSystem:
    def __init__(self):
        # 加载场景和场景的槽位
        self.load()
    def load(self):
        # 加载场景信息
        self.node_id_to_node_info = {}
        self.load_scenario('scenario-买衣服.json')
        self.load_scenario('scenario-看电影.json')
        # 加载场景槽位信息
        self.slot_info = {}
        self.load_template_slot('slot_fitting_templet.xlsx')
    # 加载场景
    def load_scenario(self, scenario_path):
        node_name = os.path.basename(scenario_path).split('.')[0]
        with open(scenario_path, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
        for node in scenarios:
            node_id = node_name + "-"+node['id']
            if "childnode" in node:
                new_child = []
                for child in node.get("childnode",[]):
                    new_child.append(node_name+"-"+child)
                node['childnode'] = new_child
            self.node_id_to_node_info[node_id] = node
    # 加载场景槽位信息
    def load_template_slot(self, template_slot_path):
        df = pandas.read_excel(template_slot_path)
        for index, row in df.iterrows():
            slot = row['slot']
            query = row['query']
            values = row['values']
            self.slot_info[slot] = [query, values]

    # 自然语言理解
    def nlu(self, memory):
        # 意图识别
        memory = self.get_intent(memory)
        # 槽位抽取
        memory = self.get_slot(memory)
        return memory
    def get_intent(self,memory):
        # 找到当前可用节点中和用户输入最接近的
        max_score = -1
        hit_intent = None
        for node_id in memory["available_node"]:
            score = self.get_node_score(node_id, memory)
            if score > max_score:
                max_score = score
                hit_intent = node_id
        memory['hit_intent'] = hit_intent
        memory['hit_intent_score'] = max_score
        return memory
    # 计算节点和用户输入的匹配度
    def get_node_score(self, node_id, memory):
        scores = []
        user_input = memory["user_input"]
        node_intents = self.node_id_to_node_info[node_id].get("intent",[])
        for node_intent in node_intents:
            # 计算得分
            score = self.get_intent_score(node_intent, user_input)
            scores.append(score)
        return max(scores)
    # jaccard距离
    def get_intent_score(self, sentence1, sentence2):
        # 计算jaccard距离
        set1 = set(sentence1)
        set2 = set(sentence2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)

    def get_slot(self, memory):
        # 槽位抽取,看用户的输入中有没有需要的槽位
        hit_intent = memory["hit_intent"]
        for slot in self.node_id_to_node_info[hit_intent].get("slot", []):
            _, values = self.slot_info[slot]
            if re.search(values, memory["user_input"]):
                memory[slot] = re.search(values, memory["user_input"]).group()
        return memory

    # 对话状态跟踪,就是看这个意图的槽位有没有填满
    def dst(self, memory):
        # 得到命中的意图
        hit_intent = memory['hit_intent']
        # 得到意图的槽位
        slots = self.node_id_to_node_info[hit_intent].get("slot", [])
        # 检查槽位是否填满
        for slot in slots:
            if slot not in memory:
                memory["need_slot"] = slot
                return memory
        memory["need_slot"] = None
        return memory
    # 对话策略,此处只是策略，具体要怎么回答，到自然语言生成中回答
    def policy(self, memory):
        # 判断当前的命中的意图槽位中是否有欠缺
        if memory['need_slot'] is None:
            memory['action'] = 'answer'
            # 没有需要的槽位了，则可以方位下一个节点了
            memory["available_node"] = self.node_id_to_node_info[memory['hit_intent']].get("childnode", [])
            # 这里在项目中会有一个执行动作的行为，去做具体的事情，例如去查询sql或者访问api
            # self.take_action(memory)
        else:
            memory['action'] = 'ask'
            # 停留在当前节点
            memory["available_node"] = [memory['hit_intent']]
        return memory

    def replace_slot(self, sentence, memory):
        hit_ident = memory['hit_intent']
        slots = self.node_id_to_node_info[hit_ident].get("slot", [])
        for slot in slots:
            sentence = sentence.replace(slot, memory[slot])
        return sentence

    # 文本生成
    def nlg(self, memory):
        if memory['action'] == 'answer':
            answer = self.node_id_to_node_info[memory['hit_intent']].get("response")
            # 此处需要一个模板替换的过程，把response中的slot给替换为正确的

            memory['bot_response'] = self.replace_slot(answer, memory)
        else:
            # 反问，看还需要哪些槽位
            slot = memory["need_slot"]
            query, _ = self.slot_info[slot]
            memory['bot_response'] = query
        return memory


    def generate_response(self, user_input, memory):
        if len(memory.get('user_input', '')) > 0:
            memory['pre'] = memory['user_input']
        if user_input in ['重听', '再说一遍', '重复一遍']:
            return memory['bot_response'], memory
        memory['user_input'] = user_input
        # 1.自然语言理解
        memory = self.nlu(memory)
        # 2.对话状态跟踪
        memory = self.dst(memory)
        # print(memory)
        # 3.对话策略
        memory = self.policy(memory)
        # 4.自然语言生成
        memory = self.nlg(memory)
        return memory['bot_response'], memory




if __name__ == '__main__':
    ds = DialogSystem()
    # print(ds.slot_info)
    # print("----------")
    # print(ds.node_id_to_node_info)
    # 状态记录，包含初始可访问节点
    memory = {"available_node": ["scenario-买衣服-node1","scenario-看电影-node1"]}
    while True:
        user_input = input('user:')
        response, memory = ds.generate_response(user_input, memory)
        print('bot:', response)


