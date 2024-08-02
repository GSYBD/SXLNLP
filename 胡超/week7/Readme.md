# 项目实现
## 1. 描述
基于电商评论数据集对不同的模型进行文本分类试验，得到各模型训练结果并统计
## 2. 具体实现
1. 添加前处理方法utils.preprocess_data_and_config,将电商数据集即CSV文件分成train_data.json和test_data.json，同时更新Config
2. 将label_to_index动态配置在config.py的Config中，以便当前代码可以更好的扩展
3. 保存各模型不同epoch的模型参数以及训练结果
4. 将同批次测试的所有模型各配置参数以及结果输出到同一个excel文件中
5. 针对excel文件做了部分优化，acc最高的特定模型所在的列整行会用绿色高亮
6. 添加了模型训练的日志，方便查看训练过程
## 3. 目录结构
```mermaid
week7
├── bert-base-chinese
├── data
│   ├── 文本分类练习.csv
│   ├── test_data.json
│   └── train_data.json
├── logs
├── output
├── reports
├── chars.txt
├── config.py
├── evaluate.py
├── loader.py
├── main.py
├── model.py
├── Readme.md
└── utils.py
```
## 4. 报告展示
详见reports/training_report_20240801234335.xlsx