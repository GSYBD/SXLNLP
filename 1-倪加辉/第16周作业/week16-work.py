"""
实现一个简单的订票小助手

"""
import json
import re

import pandas


class memory:
    def __init__(self):
        # 本轮对话命中的节点
        self.hit_node_id = {}
        # 当前对话可用节点
        self.available_node = {}
        # 节点路径
        self.node_path = ""
        # 模版路径
        self.template_path = ""
        # 还没填的槽位
        self.miss_slot = {}
        # 已经填的槽位
        self.fill_slot = {}
        # 对话策略
        self.policy = ""
        # 客户的提问
        self.query = ""
        # 每轮的回答
        self.answer = ""

    def __str__(self):
        # 打印所有字段信息
        return f"hit_node_id: {self.hit_node_id}, available_node: {self.available_node},\n " \
               f"node_path: {self.node_path}, template_path: {self.template_path}, \n" \
               f"miss_slot: {self.miss_slot}, fill_slot: {self.fill_slot},\n" \
               f" policy: {self.policy}, query: {self.query}, answer: {self.answer}\n"


class ticketAssistant:
    def __init__(self, memory):
        # 加载模版和节点数据
        self.memory = memory
        # 保存所有的node info
        self.node_list = {}  # node_id : node_info
        # 保存所有的slot info
        self.slot_list = {}  # slot : [query,value]
        self.load_data()

    def load_data(self):
        # 加载节点数据
        self.load_node_data()
        # 加载模版数据
        self.load_template_data()
        print(self.node_list)
        print(self.slot_list)

    def load_node_data(self):
        # 加载json数据
        with open(self.memory.node_path, "r", encoding="utf-8") as f:
            for node in json.load(f):
                self.node_list[node["id"]] = node
                # 如果包含node1
                if "node1" in node["id"]:
                    # 初始化默认在第一个节点
                    self.memory.available_node = [node["id"]]

        return

    def load_template_data(self):
        # 加载模版
        self.template = pandas.read_excel(self.memory.template_path)
        for index, row in self.template.iterrows():
            self.slot_list[row["slot"]] = [row["query"], row["values"]]
        return

    def run(self, query):
        self.memory.query = query

        # 意图识别 + 槽位填充
        self.nlu()
        # 对话状态
        self.dst()
        # 对话策略
        self.dpo()
        # 生成对话
        self.nlp()
        return self.memory

    def nlu(self):
        # 意图识别
        self.intention_recognition()
        # 槽位填充
        self.fit_slot()

    # 检查当前状态
    def dst(self):
        # 检查当前槽位状态是否还有没填充的
        # 获取命中节点
        hit_node_id = self.memory.hit_node_id
        # 获取槽位list
        slot_list = self.node_list[hit_node_id].get("slot", [])
        for slot in slot_list:
            if slot not in self.memory.fill_slot:
                self.memory.miss_slot = slot
                return
        # 槽位填充完毕
        self.memory.miss_slot = None
        return

    def dpo(self):
        # 根据槽位状态 选择对话策略
        if self.memory.miss_slot:
            self.memory.policy = "slot_filling"
            # 留在当前节点
            self.memory.available_node = [self.memory.hit_node_id]
        else:
            self.memory.policy = "dialogue_continue"
            # 去下个节点
            # 如果存在 childnode
            if self.node_list[self.memory.hit_node_id].get("childnode"):
                self.memory.available_node = self.node_list[self.memory.hit_node_id]["childnode"]
            else:
                # 没有childnode 对话结束
                self.memory.available_node = []
        return

    # 按照对话策略 生成返回用户的answer
    def nlp(self):
        # 槽位填充 询问获取具体槽位信息
        if self.memory.policy == "slot_filling":
            # 获取槽位信息
            slot = self.memory.miss_slot
            # 获取槽位query
            query, _ = self.slot_list[slot]
            # 生成回答
            self.memory.answer = query
        elif self.memory.policy == "dialogue_continue":
            # 获取命中节点的response
            hit_node_id = self.memory.hit_node_id
            response = self.node_list[hit_node_id]["response"]

            # 替换槽位
            slot_list = self.node_list[hit_node_id].get("slot", [])
            for slot in slot_list:
                response = response.replace(slot, self.memory.fill_slot[slot])
            self.memory.answer = response
            # 如果当前节点为回滚修改节点 需要跳转到对应节点
            if hit_node_id == "ticket-node5":
                # 对于命中节点的childnode，遍历每个slot，看改正的信息位于哪个slot
                # query_slot = self.memory.query.sub(0, self.memory.query.index("改正为"))
                # 截取 xx改正为xxx的后半部分

                query = self.memory.query
                query_slot_value = query[query.index("改为"):]
                for node in self.memory.available_node:
                    node_info = self.node_list[node]
                    slot_list = node_info.get("slot", [])
                    # 这里可以用文本匹配
                    for slot in slot_list:
                        slot = slot.replace("#", "")
                        # 回答如果命中了slot
                        if slot in self.memory.query:
                            # 修改槽位
                            self.memory.hit_node_id = node
                            self.memory.fill_slot[slot] = query_slot_value
                            self.memory.answer = node_info["response"]
                            self.memory.available_node = node_info["childnode"]
                            # 需要手动跳转到nlp
                            self.nlp()
                            return

        return

    def intention_recognition(self):
        hit_node_id = None
        hit_score = 0
        for node_id in self.memory.available_node:
            # 获取 node信息
            node_info = self.node_list[node_id]
            # 获取每个节点的intent 做匹配 返回最相似的node
            intent = node_info["intent"]
            # 计算相似度
            cal_score = self.cal_similarity(intent, self.memory.query)
            # 更新命中节点 更新命中分数
            if cal_score >= hit_score:
                hit_node_id = node_id
                hit_score = cal_score
        self.memory.hit_node_id = hit_node_id
        return

    # 使用jaccard相似度计算
    def cal_similarity(self, str1_list, str2):
        score = []
        # 可能有多个 intent
        for str1 in str1_list:
            score.append(len(set(str1) & set(str2)) / len(set(str1) | set(str2)))
        return max(score)

    # 槽位填充 根据用户的输入信息 填入槽位
    def fit_slot(self):
        # 获取命中节点
        hit_node_id = self.memory.hit_node_id
        # 获取槽位list
        slot_list = self.node_list[hit_node_id].get("slot", [])
        for slot in slot_list:
            # 不能同时命中出发和目的地
            if "出发城市" in self.memory.answer and slot == "#目的地#":
                continue
            if "目的城市" in self.memory.answer and slot == "#出发地#":
                continue
            # 对于每个槽，看用户输入信息是否包含该槽位信息
            # 获取槽的values信息
            _, values = self.slot_list[slot]
            # 搜索问题中是否含有想要的slot信息
            search_result = re.search(values, self.memory.query)
            # 如果搜到了 就保存 后面替换
            if search_result:
                self.memory.fill_slot[slot] = search_result.group()
            # else:
            # 如果没有搜到 记录下没搜到的slot
            # self.memory.miss_slot.append(slot)

        return


if __name__ == '__main__':
    memory = memory()
    memory.node_path = r"scenario/scenario-ticket-order.json"
    memory.template_path = r"scenario/slot_fitting_templet.xlsx"

    assistant = ticketAssistant(memory)
    while True:
        query = input()
        memory = assistant.run(query)
        print(memory)
        print(memory.answer)
        if "出票成功" in memory.answer:
            break
