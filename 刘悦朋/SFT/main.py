import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from loader import DataGenerator
from config import Config
from model import STFModel


def main():
    data_generator = DataGenerator(Config['train_data'])
    data_loader = DataLoader(data_generator, batch_size=Config['batch_size'], shuffle=True)

    stf_model = STFModel()
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        stf_model = stf_model.cuda()

    optimizer = Adam(stf_model.parameters(), lr=Config['learning_rate'])

    for epoch in range(Config['epoch']):
        epoch += 1
        stf_model.train()
        train_loss = []
        for idx, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, mask, labels = batch_data
            loss = stf_model(input_id, mask=mask, y=labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print('Epoch %d average loss: %f' % (epoch, np.mean(train_loss)))
        print(generate_sentence('阿根廷歹徒抢服装尺码不对拿回店里换', stf_model))


def generate_sentence(openings, model):
    tokenizer = BertTokenizer.from_pretrained(Config['bert'])
    cuda_flag = torch.cuda.is_available()
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <= 64:
            x = torch.LongTensor([openings])
            if cuda_flag:
                x = x.cuda()
            y = model(x)[0][-1]
            idx = int(torch.argmax(y))
            openings.append(idx)
    return tokenizer.decode(openings)


if __name__ == '__main__':
    main()
