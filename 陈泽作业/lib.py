import pickle

ws = pickle.load(open("./ws.pkl", "rb"))
embedding_dim = 256
max_len= 30
epochs = 20
hidden_size=128
num_layers=2
bidirectional=True
drop_out=0.2