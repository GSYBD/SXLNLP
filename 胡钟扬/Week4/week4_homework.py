import numpy as np
from collections import defaultdict


from typing import Dict, List,Optional,Literal


#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
word_dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}



#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, word_dict, win_len, all_cut_list:List[List[str]], index):
    
    '''
        必须用递归实现
        
        index: 窗口第一个字符所在字符串中的位置
    '''
    res = []
    # index=0
    begin=0 # 当前窗口取的是否是句子开头？yes：0
    
    
    # split_index=0 # 记录全切分字典的key
    # all_cut_list:Dict[int, List[str]] = defaultdict(list)
    
    if index == 0:
        pass
    
    if win_len<=1:
        window = sentence[index]
        res.append(window)
        index += win_len
    flag=0
    while index < len(sentence)-win_len+1:
        window = sentence[index: index + win_len]
        
        # 如果，该词语的长度下，字典中找不到，那就长度-1

        if window not in word_dict.keys() and index==0:
            
            # 先把win_len -1, 再把sentence还未匹配的部分传进去
            flag=1 # 代表需要缩小窗口
            all_cut(sentence[index:], word_dict, win_len-1, all_cut_list)
            break
            # res.append(all_cut(sentence[index:], word_dict, max_len-1))
        
        # 如果窗口词在词典中
        
        # 1. 让窗口再减小一格去匹配
        all_cut(sentence[index:], word_dict, win_len-1, all_cut_list, index)
        
        # 选择2： 直接切分，然后让词边界后移
        res.append(window)
        index += win_len
            
            
            
        if flag == 1:
            continue
        else:
            print("============= 切分结束 =====================")
            res = "/".join(res)
            print(res)
        
        
    return res
    
    
    
    
# all_cut_list = []
    
# all_cut(sentence, word_dict, max([len(k) for k in word_dict.keys()]), all_cut_list, 0)



def all_cut2(sentence, word_dict, max_len):
    '''
    先做一个普通的正向切分
    
    
    返回最后一个词的窗口长度
    '''
    result = [] # 切分结果
    last_win_len=0
    while sentence != "":
        win_len =min(len(sentence), max_len)
        window  = sentence[:win_len]
        
        
        while window not in word_dict:
            if len(window)==1:
                break
            window = window[:len(window)-1]
        sentence = sentence[len(window):]

        result.append(window)
        
        if sentence =="":
            last_win_len = len(window)
        
    return result, last_win_len
            


# result = all_cut2(sentence, word_dict,3)

# print(result)


def all_cut3(sentence, word_dict):
    final_result: List[List[str]] = []  # 嵌套列表，用来存放所有的切分结果
    from copy import deepcopy
    
    sentence_copy=deepcopy(sentence)
    
    max_len = max([len(key) for key in word_dict.keys()])
    

    # 从最后一个词开始倒退窗口
    result,last_win_len = all_cut2(sentence, word_dict, max_len)
    
    print("result = ",result)
    print("====================")
    prefix= result[:-1]
    

    suffix = [result[-1]]
    # final_result.append(result)  
    
    index = len(result)-1
    # 外部循环用来倒退suffix窗口
    
               #  ['经常', '有意见', '分歧'],

    while prefix!=[]:
        print("prefix = ",prefix)
        print("suffix = ", suffix)
        print("==========================")
        loop_win_len =len(suffix[0])-1
        
        final_result.append(prefix+suffix)  
        
        
        suffix = "".join(suffix) # "有意见分歧"
        
        inner_prefix = prefix
        
        # while len(inner_window_prefix) > len(prefix)
        
        
        while len(inner_prefix)>=len(prefix): # 窗口内倒退不能超过prefix的边界
            last_split = []
            # 内循环用来切分suffix窗口中的词
            # 递归切分suffix, 直到suffix的最后一个切分出来的词是词表中的最短词（无法再被切分）
            while is_splitable(suffix, word_dict): # suffix可以再被切分
                
                suffix_str = "".join(suffix)
                suffix_split, tmp_win_len =  all_cut2(suffix_str, word_dict, loop_win_len) 
                
                print("suffix_split = ", suffix_split)
                print("tmp_win_len = ", tmp_win_len)
                print("==============================")
                
                
                loop_win_len = tmp_win_len-1  # 下一次切分的窗口长度取 len(suffix_split)-1, 如果不-1的话，还会切分出同一个词
                
                suffix_prefix= suffix_split[:-1]
                
                if isinstance(suffix_prefix, str):
                    suffix_prefix = [suffix_prefix]
                    
                suffix_suffix = [suffix_split[-1]]
                
                print("suffix_prefix = ",suffix_prefix)
                print("suffix_suffix = ",suffix_suffix)
                print("===============================")
                
                # 先保存当前的切分
                final_result.append(inner_prefix+suffix_split)
                
                
                # 保存最后一轮的切分结果
                last_split = inner_prefix+suffix_split
                
                # 更新prefix
                inner_prefix+=suffix_prefix
                
                # 更新suffix
                suffix = suffix_suffix
                
                suffix = "".join(suffix)
                
            
            
            # 接着进行细粒度的倒退(窗口内倒退)：在当前的切分下，倒退一个窗口， 单个字不算窗口
            # 并且窗口内倒退的范围不能超过prefix窗口的切分范围
            small_backward_index=0
            for i  in range(0, len(last_split),-1):
                if len(last_split[i])>1 and i!=len(last_split)-1: # 从后往前找第一个不是单字的窗口, 并且最后一个窗口suffix我们已经确保它被切完了，因此不用判断
                    # 修改 inner_prefix 和 suffix
                    inner_prefix  = inner_prefix[:i+1]
                    suffix = inner_prefix[i+1:]
                    suffix - "".join(suffix)
                    
                    break
                    
            
        
        # [经常, 有意见, 所以分歧大, 很正常]
        
        
        
        
        # 后缀已经无法再被切分，出循环
        
        # 将prefix部分进行回退 (粗粒度倒退/窗口级倒退)
        index-=1
        prefix= result[:index]
        suffix = result[index:]
        # mid = len(prefix[:-1])
        # prefix = "".join(result)[:mid]
        # suffix = "".join(result)[mid:]
        
    return final_result
        



def inner_splitter(prefix:str, suffix:str, word_dict:dict):
    
    '''
      实现对后缀窗口内的全切分
    
    '''
    
    last_split = []
    while is_splitable(suffix, word_dict):
            
            suffix_str = "".join(suffix)
            suffix_split, tmp_win_len =  all_cut2(suffix_str, word_dict, loop_win_len) 
            
            print("suffix_split = ", suffix_split)
            print("tmp_win_len = ", tmp_win_len)
            print("==============================")
            
            
            loop_win_len = tmp_win_len-1  # 下一次切分的窗口长度取 len(suffix_split)-1, 如果不-1的话，还会切分出同一个词
            
            suffix_prefix= suffix_split[:-1]
            
            if isinstance(suffix_prefix, str):
                suffix_prefix = [suffix_prefix]
                
            suffix_suffix = [suffix_split[-1]]
            
            print("suffix_prefix = ",suffix_prefix)
            print("suffix_suffix = ",suffix_suffix)
            print("===============================")
            
            # 先保存当前的切分
            final_result.append(prefix+suffix_split)
            
            last_split = prefix+suffix_split
            
            # 更新prefix
            prefix+=suffix_prefix
            
            # 更新suffix
            suffix = suffix_suffix
            
            suffix = "".join(suffix)
            
    return last_split  
            
            
def is_splitable(sentence:str, word_dict):
     '''
        根据词表判断该句子是否可再被切分
     '''       
     
     for key in word_dict.keys():
         if key in sentence and len(key)< len(sentence):
             return True
    
     return False
 
 
 
def print_split(result:List[List[str]]):
     
     for i in range(len(result)):
         print(result[i])
        

#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]







if __name__ == '__main__':
    final_result = all_cut3(sentence,word_dict)


    print_split(final_result)


