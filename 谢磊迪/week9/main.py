from model import Ner_Model, choice_optim
from create_datas import load_datas, config
from evaluate import Evaluate


def main(train_or_test_path):
    dl = load_datas(config, train_or_test_path)
    print(config)
    print("ok")
    model = Ner_Model(config)
    optim = choice_optim(model, config)
    model.train()
    evaluate = Evaluate(config, train_or_test_path="test_path")
    for epoch in range(config["epochs"]):
        for batch_data in dl:
            x = batch_data[0]
            y = batch_data[1]
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
        print(f"{epoch} 的损失函数是{loss}")
        evaluate.evaluate(model)


if __name__ == "__main__":
    train_or_test_path = "train_path"
    main(train_or_test_path)
