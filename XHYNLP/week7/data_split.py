import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    # 读取CSV文件
    data = pd.read_csv('F:\\nlp\\week7\\文本分类练习.csv')
    
    # 检查数据
    print("Data Shape:", data.shape)
    
    # 划分数据
    train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # 转换为JSON格式
    train_json = train_data.to_dict(orient='records')
    valid_json = valid_data.to_dict(orient='records')
    
    # 保存到文件
    save_json(train_json, 'F:\\nlp\\week7\\train.json')
    save_json(valid_json, 'F:\\nlp\\week7\\valid.json')

if __name__ == "__main__":
    main()