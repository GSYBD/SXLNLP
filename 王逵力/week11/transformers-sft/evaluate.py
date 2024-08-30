import torch

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger

    def eval(self, epoch):
        self.model.eval()
        # 通常会加载验证数据或测试数据，这里假设我们有验证集的数据
        # validation_data = load_validation_data()  # 示例函数

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in validation_data:
                input_ids, attention_mask, labels = batch_data
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    labels = labels.cuda()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        self.logger.info(f"Epoch {epoch} 验证准确率: {accuracy:.4f}")