# 项目实现
## 1. 描述
使用Bert实现NER(即命名实体识别)
## 2. 具体实现
1. config文件添加'model_type'，以便控制使用的模型（即控制损失函数）以及DataGenerator类中的数据生成方式
2. config文件中num_layers配置为3
3. 当model_type为bert时，
    * model.py中使用BertModel，并使用BertEmbedding
    * ```self.bert.config.num_hidden_layers = num_layers # 即层数修改为3```
    * loader.py中使用BertTokenizer