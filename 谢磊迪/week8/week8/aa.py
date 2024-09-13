from  model import *



import random

a4 = [[['a']], [['b','5']], [['c']], ['b','5','d'], ['e']]  # 示例列表
sentens_li = a4.copy()
first_choice = random.sample(sentens_li, 1)  # 第一次随机选择
sentens_li.remove(first_choice[0])  # 从列表中移除选中的元素
second_choice = random.sample(sentens_li, 1)  # 第二次随机选择
sentens_li.remove(second_choice[0])  # 从列表中移除选中的元素
three_choice = random.sample(sentens_li, 1)  # 第二次随机选择
print(1)