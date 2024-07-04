def cross_entropy(y_true,y_pred):
    C=0
    # one-hot encoding
    for col in range(y_true.shape[-1]):
        y_pred[col] = y_pred[col] if y_pred[col] < 1 else 0.99999
        y_pred[col] = y_pred[col] if y_pred[col] > 0 else 0.00001
        C+=y_true[col]*np.log(y_pred[col])+(1-y_true[col])*np.log(1-y_pred[col])
    return -C

# 没有考虑样本个数 默认=1
num_classes = 3
label=1#设定是哪个类别 真实值

y_true = np.zeros((num_classes))
# y_pred = np.zeros((num_classes))
# preset
y_true[label]=1
y_pred = np.array([0.0,1.0,0.0])
C = cross_entropy(y_true,y_pred)
print(y_true,y_pred,"loss:",C)
y_pred = np.array([0.1,0.8,0.1])
C = cross_entropy(y_true,y_pred)
print(y_true,y_pred,"loss:",C)
y_pred = np.array([0.2,0.6,0.2])
C = cross_entropy(y_true,y_pred)
print(y_true,y_pred,"loss:",C)
y_pred = np.array([0.3,0.4,0.3])
C = cross_entropy(y_true,y_pred)
print(y_true,y_pred,"loss:",C)

