import os

def merge_files(folder_path, output_file):
    # 创建一个空字符串用于存储合并的内容
    merged_content = ""
    
    # 获取指定文件夹下的所有.txt文件名
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # 按文件名排序
    txt_files.sort()
    
    # 遍历每个.txt文件
    for file_name in txt_files:
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            content = file.read()
            # 在每个文件内容之间添加分割符
            merged_content += content + '\n\n'
    
    # 写入新的.txt文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(merged_content)
    
    return merged_content

folder_path = 'week14\Heroes'
output_file = 'week14\merge_h.txt'

# 合并文件并获取合并后的内容
merged_text = merge_files(folder_path, output_file)

# 打印合并后的文本内容
print(merged_text)