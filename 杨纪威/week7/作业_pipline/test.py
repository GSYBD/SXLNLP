import csv

class YourCSVProcessor:
    def __init__(self, file_path):
        self.path = file_path

    def process_csv(self):
        data_list = []
        with open(self.path, mode='r', encoding='utf-8', newline='') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # 读取CSV文件的第一行作为header
            for row in csv_reader:
                # row即为每一行的数据
                print("line:", row)

# 假设有一个名为 "train.csv" 的CSV文件，包含数据行
file_path = "../作业_data/train.csv"

# 创建一个处理CSV文件的实例
csv_processor = YourCSVProcessor(file_path)

# 处理CSV文件
csv_processor.process_csv()



import csv
from io import StringIO

# 两行数据的CSV字符串
csv_data = '''1,"肉卷吃的我有点腻,个人口味的问题吧,皮蛋瘦肉粥不错"
1,太棒了，速度快，还好吃，只是粥再多一点就好了，希望采纳'''

# 使用StringIO将字符串转换为类文件对象
csv_file = StringIO(csv_data)

# 创建一个CSV reader
csv_reader = csv.reader(csv_file)
print("csv_reader",csv_reader)
# 遍历每行并打印每行数据的列表表示
for row in csv_reader:
    print(row)
