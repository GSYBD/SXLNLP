import json  
import random  

def split_json(json_file_path, train_file_path, val_file_path, train_ratio=0.8):  
    # 读取JSON文件  
    with open(json_file_path, mode='r', encoding='utf-8') as json_file:  
        data = [json.loads(line) for line in json_file  ]
    
    # 打乱数据  
    random.shuffle(data)  
    
    # 计算训练集的大小  
    train_size = int(len(data) * train_ratio)  
    
    # 分割数据  
    train_data = data[:train_size]  
    val_data = data[train_size:]  
    
    # 保存训练集  
    with open(train_file_path, mode='w', encoding='utf-8') as train_file: 
        for item in train_data: 
            json_line = json.dumps(item, ensure_ascii=False)  
            train_file.write(json_line+"\n")
    # 保存验证集  
    with open(val_file_path, mode='w', encoding='utf-8') as val_file:
        for item in val_data:
            json_line = json.dumps(item, ensure_ascii=False)  
            val_file.write(json_line+"\n") 

# 示例调用  
split_json(r'week7_homework\data\e_market_comment.json', r'week7_homework\data\train_data.json', r'week7_homework\data\valid_data.json', train_ratio=0.8)
print("分割成功~~~")