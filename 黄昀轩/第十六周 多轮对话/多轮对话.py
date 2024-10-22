
import re 
import json
import pandas as pd
import os


'''
基于脚本的多轮对话
'''

class DialogueSys():
    def __init__(self):
        self.load()

    def load(self):
        #加载场景
        self.load_node_id_to_node_info={} #创建一个空字典存储各个node #数据结构存在外面
        self.load_scenario('./scenario/scenario-买衣服.json')
        self.load_scenario('./scenario/scenario-看电影.json')
        #加载槽位模板
        
        self.slot_info = {} #数据结构存在外面
        self.load_slot_template('./scenario/slot_fitting_templet.xlsx')

    #载入json 场景    
    def load_scenario(self,scenario_file):
        scenario_name = os.path.basename(scenario_file.split('.')[1])
        with open(scenario_file,'r',encoding='utf-8') as f:
            self.scenario = json.load(f)
        for node in self.scenario:
            #拼接文件名
            node_id = scenario_file + '_' + node['id'] #从for 循环中读取的单个node中获取id,并添加到场景文件名后面***
            if 'childnode' in node:
                new_child = []
                #把childnode中的node_id拼接上文件名放进new_childnode列表中
                for child in node['childnode']:
                    child_id = scenario_file + '_' + child
                    new_child.append(child_id)
                #把原来的['childnode']替换成新的new_childnode值
                node['childnode'] = new_child
            #把调整后的单个node添加到self.load_node_id_to_node_info字典中，进入下一次的循环，加载新node
            #往之前的node字典中添加node_id和node的键值对
            self.load_node_id_to_node_info[node_id] = node 
        print('场景加载成功')

    #载入槽位模板
    def load_slot_template(self,slot_template_file):
        df = pd.read_excel(slot_template_file)
        for index,row in df.iterrows():
            slot = row['slot']
            query = row['query']
            values = row['values']
            self.slot_info[slot] = [query,values]
        return 

    def nlu(self,memory):
        # 意图识别 
        memory = self.get_intent(memory)
        # 槽位抽取
        memory = self.get_slot(memory)
        return memory
    
    def get_intent(self,memory):
        #设置初始得分
        max_score = -1  
        hit_intent = None
        for node_id in memory['avalaible_nodes']:
            node = self.load_node_id_to_node_info[node_id]
            #计算当前node方法在外面单独写
            score = self.get_node_score(node,memory) 
            #只有循环中 当前node得分大于max_score才会覆盖，否则跳过
            if score > max_score:
                #比较后重新赋值max_score
                max_score = score 
                #记录当前的得分后的node_id
                hit_intent = node_id
        #遍历完成后，把hit_intent 赋值到memory中
        memory['hit_intent'] = hit_intent #记录得分最大的id
        memory['hit_intent_score'] = max_score
        return memory

    def get_node_score(self,node,memory):
        #单个节点计算得分
        intent = memory['user_input']
        node_intents = node['intent']
        scores = []
        #针对可能有多个初始node 对各自的intent 做相似度计算的得分 取最大
        for node_intent in node_intents:
            sentence_similarity = self.get_sentence_similarity(intent,node_intent+'再说一遍'+'重复'+'什么？') #计算相似度的node_intent 里面增加'重复 再等字样' 匹配到后重重此节点
            scores.append(sentence_similarity)
        return max(scores)

    def get_sentence_similarity(self,sentence1,sentence2):
        #计算两个句子的相似度 bert jaccard 都行
        #jaccard
        set1 = set(sentence1) # 句子1拆成单词
        set2 = set(sentence2) # 句子2拆成单词
        intersection = set1.intersection(set2) #取单词的交集
        union = set1.union(set2) #取单词的并集
        return len(intersection)/len(union) #交集长度/并集长度
    
    def get_slot(self,memory):
        #槽位抽取
        #根据memory中的hit_intnent获取node_id 
        hit_intent = memory['hit_intent'] 
        #for循环是循环node_info的slot中的多个槽位
        for slot in self.load_node_id_to_node_info[hit_intent].get('slot',[]):
            #根据slot，转到slot_info获取query和values
            _,values = self.slot_info[slot]
            if re.search(values,memory['user_input']) :
                #这里是memory[slot]直接创建取到的slot 创建键值对
                memory[slot] = re.search(values,memory['user_input']).group()
        return memory
    





    #dst对话状态跟踪 （槽位有没有被填满）
    def dst(self,memory):
        hit_intent = memory['hit_intent']
        slots =  self.load_node_id_to_node_info[hit_intent].get('slot',[])
        for slot in slots:
            if slot not in memory:
                memory['need_slot'] = slot
                return memory #这里循环检测slot，如果检测到有未填的slot，就返回memory，并结束函数，直接到policy区ask完成后再循环回来，再检测其他的slot
        memory['need_slot'] = None #上面全部检测完了，把memory['need_slot']置为None,返回
        #这里是向下传递need_slot的状态
        return memory  


    #policy对话策略 根据need_slot情况 决定action的状态
    def policy(self,memory):
        #如果槽位有欠缺，反问
        #如果没有欠缺，直接回答
        if memory['need_slot'] is None:
            memory['action'] = 'answer'
            #开放子节点
            memory['avalaible_nodes'] = self.load_node_id_to_node_info[memory['hit_intent']].get('childnode',[])
            #行为动作（预留位置）
 
        else:
            memory['action'] = 'ask'
            #停留在父节点
            memory['avalaible_nodes'] = [memory['hit_intent']]
        
        return memory
    #这里定义nlg的的一个方法  
    def replace_slot(self,text,memory):
        #替换responde中的槽位
        hit_intent = memory["hit_intent"] #注意这里hit_intent里面是节点路径
        slots = self.load_node_id_to_node_info[hit_intent].get('slot',[])  #注意这里hit_intent里面是节点路径#注意着get是括号
        for slot in slots:
            text= text.replace( slot,memory[slot]) 
            #把responde中的slot（键的字符）替换成memory中slot对应值的字符，实现槽位替换
        return text
            



    #nlg根据回答的状态，生成回复
    def nlg(self,memory):
        #直接回答
        if memory['action'] == 'answer':
            answer = self.load_node_id_to_node_info[memory['hit_intent']]['response']
            memory['bot_response'] = self.replace_slot(answer,memory) #替换responde中的槽位
        #反问问题
        else:
            slot= memory['need_slot']
            print(slot)
            #再slot_info中找到对应的query和values
            query,values = self.slot_info[slot]
            memory['bot_response'] = query
        return memory


    def generate_response(self,user_input,memory):
        #记录用户输入
        memory['user_input'] = user_input
        memory = self.nlu(memory)
        # print('memo==>',memory)
        #用户跟踪
        memory = self.dst(memory)
        #对话策略
        memory = self.policy(memory)
        #生成回复
        response = self.nlg(memory)
        return memory['bot_response'],memory





if __name__ == '__main__':
    ds = DialogueSys()
    print('node_info===>',ds.load_node_id_to_node_info)
    print('slot_info===>',ds.slot_info)
    memory = {"avalaible_nodes":['./scenario/scenario-买衣服.json_node1','./scenario/scenario-看电影.json_node1']}
    #多轮对话，采用while循环来实现
    while True:
        user_input = input('user:')#获取用户输入
        response,memory = ds.generate_response(user_input,memory)#调用生成回复的函数
        print('bot:',response)#打印回复

        