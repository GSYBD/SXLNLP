import random
"""切割训练集和验证集"""
def split_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]
    random.shuffle(lines)
    num_lines = len(lines)
    num_train = int(0.8 * num_lines)
    train_lines = lines[:num_train]
    valid_lines = lines[num_lines:]
    with open('train_data.txt', 'w', encoding='utf8') as f_train:
        f_train.writelines(train_lines)
    with open('valid_data.txt', 'w', encoding='utf8') as f_valid:
        f_valid.writelines(valid_lines)

split_file(r'E:\software\PyCharm\pythonProject\.venv\Learning\week7\文本分类练习数据集\文本分类练习.csv')
