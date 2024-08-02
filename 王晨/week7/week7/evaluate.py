import torch

def evaluate(model, x_test, y_test):
    model.eval()
    print(f'正样本数量{sum(y_test)},负样本数量{len(y_test)}')
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x_test)
        # print(y_pred)
        for y_p, y_t in zip(y_pred, y_test):
            if torch.argmax(y_p) >= 0.5 and y_t == 1:
                correct += 1
            elif torch.argmax(y_p) < 0.5 and y_t == 0:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
