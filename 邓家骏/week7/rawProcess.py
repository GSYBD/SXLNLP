# 统计，按统计结果分train,valid两个文件输出
'''
1. 读一次，统计正负样本数，文本平均长度，最大长度，长度频率。用两个临时文件分别存储正样本，负样本
2. 按比例将正负样本分成train，valid
统计：
正样本数，负样本数，长度均值，每个文本长度样本数
'''

import re
import os
def write(path,context):
    with open(path,'a',encoding='utf8') as f:
        f.write(context)




def stat(path,temp_tag0_path,temp_tag1_path,temp_unk_path):
    count_dict = {'0':0,'1':0,'total':0}
    sentences_len_dict = {}
    total_len = 0
    with open(path,encoding='utf8') as f :
        next(f)
        for line in f :
            try :
                tag,s = line.split(',',1)
            except Exception as e:
                print(line,e)
            s = re.sub(r'^"|"$', '', s)
            line = tag + ',' + s
            count_dict[tag] += 1
            count_dict['total'] += 1
            s_len = len(s)
            total_len += s_len
            if s_len in sentences_len_dict:
                sentences_len_dict[s_len] += 1
            else :
                sentences_len_dict[s_len] = 1
            if(tag == '0'):
                write(temp_tag0_path,line)
            elif(tag == '1'):
                write(temp_tag1_path,line)
            else:
                write(temp_unk_path,line)
        return count_dict,sentences_len_dict,total_len

def train_vaild_split(div,i_path,train_path,vaild_path):
        with open(i_path,encoding='utf8') as input,\
        open(train_path,'a',encoding='utf8') as train_output,\
        open(vaild_path,'a',encoding='utf8') as vaild_output:
            for i,line in enumerate(input):
                if(i % div != 0):
                    train_output.write(line)
                else:
                    vaild_output.write(line)

if __name__ == "__main__":
    path = r'D:\code\data\week7_data\文本分类练习.csv'
    temp_tag0_path = r'D:\code\data\week7_data\temp_tag0.csv'
    temp_tag1_path = r'D:\code\data\week7_data\temp_tag1.csv'
    temp_unk_path = r'D:\code\data\week7_data\unk.csv'
    train_path = r'D:\code\data\week7_data\train.csv'
    vaild_path = r'D:\code\data\week7_data\vaild.csv'
    
    count_dict,sentences_len_dict,total_len = stat(path,temp_tag0_path,temp_tag1_path,temp_unk_path)
    # tag0
    div = int(count_dict['0'] / int(count_dict['0']*0.1))
    train_vaild_split(div,temp_tag0_path,train_path,vaild_path)
    if(os.path.exists(temp_tag0_path)):
        os.remove(temp_tag0_path)
    # tag1
    div = int(count_dict['1'] / int(count_dict['1']*0.1))
    train_vaild_split(div,temp_tag1_path,train_path,vaild_path)
    if(os.path.exists(temp_tag1_path)):
        os.remove(temp_tag1_path)
    
    # 输出stat
    with open('D:\code\data\week7_data\stat.csv','a',encoding='utf8') as f:
        for k,v in count_dict.items():
            f.write(f"{k},{v}\n")
        for k,v in sentences_len_dict.items():
            f.write(f"{k},{v}\n")
            # 其实没意义啊？因为这里算的是字数总长，分词后按每个句子多少个词算token长度
        f.write(f"文本总长,{total_len}\n")

    
    