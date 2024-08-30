import json
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer
from config import Config

# tittlearr = []
# contentarr = []
# with open(r"D:\code\data\week11_data\sample_data.json", encoding="utf8") as f:
#     for line in f:
#         sample = json.loads(line)
#         tittlearr.append(len(sample['title']))
#         contentarr.append(len(sample['content']))


# print(np.max(tittlearr),np.mean(tittlearr),np.median(tittlearr))
# print(np.max(contentarr),np.mean(contentarr),np.median(contentarr))

# arr = np.arange(9)
# print(arr[:2])

sentence_len = 20
ask_len = 5
mask = np.zeros((30,30),dtype=int)
for i in range(sentence_len):
    if (i < ask_len):
        mask[:sentence_len,i] = 1
    else:
        mask[i:sentence_len,i] = 1
print(mask)
# sample = json.loads('{"title": "阿根廷歹徒抢服装尺码不对拿回店里换", "content": "阿根廷布宜诺斯艾利斯省奇尔梅斯市一服装店，8个月内被抢了三次。最后被抢劫的经历，更是直接让老板心理崩溃：歹徒在抢完不久后发现衣服“抢错了尺码”，理直气壮地拿着衣服到店里换，老板又不敢声张，只好忍气吞声。（中国新闻网） "}')
# print(sample['title'],sample['content'])
tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])

print(tokenizer.decode([4500, 1166, 782, 4638, 1305, 1357, 7178, 3221, 2990, 7008, 6820, 3221, 982, 4668, 8043, 3219]))
print(tokenizer.decode([101, 3696, 2125, 123, 702, 3299, 6901, 1283, 865, 7167, 4403, 2486, 6159, 1140, 2157, 782, 1391, 7649, 2785, 2128, 1059, 2384]))
print(tokenizer.encode(' '))
print(tokenizer.decode([711]))