import numpy as np
import re

arr = np.arange(8).reshape(2,2,2)
if None:
    print('y')
else:
    print('n')

match = re.search(r"learning_rate_(.*?).pth","num_layer_3-epoch_25-use_crf_False-learning_rate_0.0001.pth")
print(float(match.group(1)))