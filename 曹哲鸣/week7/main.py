import csv
import os.path
import numpy as np
import torch.cuda
import logging
from loader import load_data
from model import Module, choose_optimizer
from evaluate import Evaluator
from config import config
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config)
    model = Module(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    optim = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optim.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optim.step()

            train_loss.append(loss.item())
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return acc

if __name__ == "__main__":
    main(config)

    header = ["Module", "lr", "hidden_size", "batch_size", "pooling", "acc"]
    output_path = "output.csv"

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model in ["rnn", "lstm", "gru", "bert", "bert_lstm"]:
            config["model_type"] = model
            for lr in [1e-3, 1e-4]:
                config["lr"] = lr
                for hidden_size in [128]:
                    config["hidden_size"] = hidden_size
                    for batch_size in [64, 128]:
                        config["batch_size"] = batch_size
                        for pooling_style in ["avg", "max"]:
                            config["pooling_style"] = pooling_style
                            acc = main(config)
                            writer.writerow([model, lr, hidden_size, batch_size, pooling_style, acc])
