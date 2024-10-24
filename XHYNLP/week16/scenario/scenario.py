import re
import os
import json
import pandas as pd

class Scenario:
    def __init__(self):
        self.load()
        
    def load(self):
        self.all_nodes_info = {}
        # 加载场景数据
        self.load_scenario("week16\scenario\scenario-买衣服.json")
        self.slot_info = {}
        # 加载槽位数据
        self.load_template("week16\scenario\slot_fitting_templet.xlsx")

    def load_scenario(self, file_path):
        scenario_name = os.path.splitext(os.path.basename(file_path))[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            for node_info in json.load(f):
                node_id = node_info['id']
                node_id = scenario_name + '_' + node_id
                if "childnode" in node_info:
                    node_info['childnode'] = [scenario_name + '_' + childnode for childnode in node_info['childnode']]
                self.all_nodes_info[node_id] = node_info
        print(f"场景 {scenario_name} 加载完成")

    def load_template(self, file_path):
        self.slot_template = pd.read_excel(file_path)
        for _, row in self.slot_template.iterrows():
            self.slot_info[row['slot']]=[row['query'],row['values']]
        print(f"槽位加载完成")
        
    def run(self, user_query, memory):
        if len(memory)==0:
            memory['avaliable_node'] = ['scenario-买衣服_node1','scenario-买衣服_node5']
            memory['last_node'] = 'scenario-买衣服_node1'
        memory['user_query'] = user_query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        print(memory)
        return memory
    
    def nlu(self, memory):
        #意图识别
        memory = self.intent_recognition(memory)
        #槽位抽取
        memory = self.slot_extraction(memory)
        
        return memory
    
    def intent_recognition(self, memory):
        #根据用户query识别意图
        # memory['avaliable_node'] = []
        #用循环与场景意图进行匹配，选择最匹配的意图
        max_score = 0
        memory['error'] = False
        if 'cur_node' in memory:
            memory['last_node'] = memory['cur_node'] 
        # cur_node_id =None
        # cur_node_info =None
        # score = 0
        for node_id in memory['avaliable_node']:
            node_info = self.all_nodes_info[node_id]
            score = self.get_intnet_score(node_info, memory)
            err_score = self.get_err_score(memory)
            if err_score >score:
                memory['error'] = True
            if score>=max_score:
                max_score = score
                memory['cur_node'] = node_id
                memory['cur_node_info'] = node_info
                memory['cur_score'] = max_score
                memory['error_score'] = err_score
            #如果为节点行为等于重复，载入last_node
            memory['action'] = self.all_nodes_info[memory['cur_node']].get('action', [])
            if memory['action'] == ["REPET"]:
                memory['cur_node']=memory['last_node']
                memory['cur_node_info']=self.all_nodes_info[memory['cur_node']]
        #         cur_node_id = node_id
        #         cur_node_info = node_info
        # memory['cur_node'] = cur_node_id
        # memory['cur_node_info'] = cur_node_info
        # memory['cur_score'] = score
        return memory
    
    def get_err_score(self, memory):
        user_query = memory['user_query']
        intent = 'error'
        score = self.get_string_score(user_query, intent)
        return score
    def get_intnet_score(self, node_info, memory):
        #计算用户query与场景意图的匹配程度
        user_query = memory['user_query']
        intent_list = node_info['intent']
        #字符串匹配选分值最高的意图
        all_score = []
        for intent in intent_list:
            score = self.get_string_score(user_query, intent)
            all_score.append(score)
        return max(all_score)
    
    def get_string_score(self, string1, string2):
        #使用jaccard相似度计算字符串匹配程度
        score =len(set(string1)&set(string2))/len(set(string1)|set(string2))
        return score
        
    def slot_extraction(self, memory):
        #根据用户query抽取槽位
        user_query = memory['user_query']
        slot_list = self.all_nodes_info[memory['cur_node']].get('slot', [])
        for slot in slot_list:
            _,candidate = self.slot_info[slot]
            search_result =  re.search(candidate, user_query)
            if search_result is not None:
                memory[slot] = search_result.group()
        return memory
    
    def dst(self, memory):
        #检查当前节点的槽位信息是否齐全
        for slot in self.all_nodes_info[memory['cur_node']].get('slot', []):
            if slot not in memory or memory[slot] == '':
                memory['missing_solt']=slot
                return memory
        memory['missing_solt']= None
        return memory
    def dpo(self, memory):
        #如果当前阶段solt信息齐全，则进入下一阶段,如果缺则进行反问
        if memory['missing_solt'] is None:
            memory['policy'] = "answer"
            memory['avaliable_node'] = self.all_nodes_info[memory['cur_node']].get('childnode', [])
        else:
            slot = memory['missing_solt']
            memory['policy'] = "ask"
            memory['avaliable_node'] = [memory['cur_node'],'scenario-买衣服_node5']
        return memory
    def nlg(self, memory):
        if memory['error'] == True:
            memory['response'] = '对不起，我无法理解您的意思，请重新描述'
            return memory
            # memory['error'] = 'false'
        #根据policy和当前节点生成回复
        if memory['policy'] == "ask":
            #memory['response'] = self.all_nodes_info[memory['cur_node']]['ask'][memory['missing_solt']]
            slot = memory['missing_solt']
            ask_sentence,_ = self.slot_info[slot]
            memory['response'] = ask_sentence
        else:
            response = self.all_nodes_info[memory['cur_node']]['response']
            #把response中的slot替换为memory中的slot
            response = self.replace_slot(response, memory)
            memory['response'] = response
        return memory
    def replace_slot(self, response, memory):
        slots = self.all_nodes_info[memory['cur_node']].get('slot',[])
        for slot in slots:
            if slot in memory:
                response = re.sub(slot, memory[slot], response)
        return response
if __name__ == '__main__':
    scenario = Scenario()
    memory = {}
    while True:
        user_query = input("User: ")    
        memory = scenario.run(user_query, memory)
    
