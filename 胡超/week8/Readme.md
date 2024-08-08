# 项目实现
## 1. 描述
使用三元组损失函数训练智能问答模型
## 2. 具体实现
1. config文件添加'model_type': 'TRIPLET'，以便控制使用的模型（即控制损失函数）以及DataGenerator类中的数据生成方式
2. 将SiameseNetwork类提为抽象类，
    * 并将loss作为抽象属性，以便强制子类实现并可自定义自己的损失函数
    * 将forward方法改为抽象方法，以便强制子类实现并可自定义自己的前向传播过程
3. 实现SiameseNetworkCEL类，继承自SiameseNetwork类，并实现loss和forward方法，loss使用CosineEmbeddingLoss
4. 实现SiameseNetworkTriplet类，继承自SiameseNetwork类，并实现loss和forward方法，loss使用cosine_triplet_loss
5. 修改DataGenerator类，使其能根据model_type动态选择对应的训练数据的生成方式