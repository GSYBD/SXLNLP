loader文件中用bert自带的tokenizer进行embedding，然后将cls与esp token的开关设为False。这样就不需要修改后续的evaluation文件。


训练时，5层一下的bert性能较低，在0.65左右，而且线性层不设激活函数性能会更低（0.55左右）。

这里bert层数设置的为6，性能最高能到0.70左右，高于BiLSTM。

而且学习率不能调太大，太大会不收敛。
