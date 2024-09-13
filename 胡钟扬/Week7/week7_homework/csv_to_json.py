import json

import csv
from typing import List, Dict
'''
本文件用于 csv, json, 字典列表 之间的转换操作
'''

def csv_to_json(csv_file_path, json_file_path):  
    # 打开CSV文件进行读取  
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:  
        # 使用DictReader读取CSV文件  
        csv_reader = csv.DictReader(csv_file)  
        
        # 将CSV数据转换为字典列表  
        # data = [row for row in csv_reader]  
    
        # 将字典列表转换为JSON格式  
        with open(json_file_path, mode='w', encoding='utf-8') as json_file: 
            for row in csv_reader:
                # 将每一行字典转换为JSON字符串，并写入文件，逐行写  
                json_line = json.dumps(row, ensure_ascii=False) 
                json_file.write(json_line + "\n") 



def write_dicts_to_csv(source:List[Dict], target_path:str):
    '''
        将字典列表写入csv文件
    '''
    
    # 打开文件进行写入  
    with open(target_path, mode='w', newline='', encoding='utf-8') as file:  
        # 创建CSV字典写入器  
        writer = csv.DictWriter(file, fieldnames=source[0].keys())  
        
        # 写入表头  
        writer.writeheader()  
        
        # 写入数据行  
        for row in source:  
            writer.writerow(row)  

    print(f"Data has been written to {target_path}") 
    



if __name__ == '__main__':
    # 示例调用  
    csv_to_json(r'week7_homework\data\e_market_comment.csv', r'week7_homework\data\e_market_comment.json')  
    print("解析完毕~")