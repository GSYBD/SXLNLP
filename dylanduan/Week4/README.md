第四周作业

实现全切分函数，输出根据字典能够切分出的所有的切分方式

```
def all_cut(sentence, Dict):
    def recursive_cut(sub_sentence):
        results = []
        for i in range(1, len(sub_sentence) + 1): # 循环递归
            prefix = sub_sentence[:i]       # 前缀
            if prefix in Dict:
                if i == len(sub_sentence):      # 到达末尾
                    results.append([prefix])
                else:
                    # 否则对剩余部分继续递归切分
                    for suffix_cut in recursive_cut(sub_sentence[i:]):
                        results.append([prefix] + suffix_cut)
        return results
    
    return recursive_cut(sentence)

result = all_cut(sentence, Dict)
```
