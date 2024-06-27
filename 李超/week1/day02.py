from __future__ import print_function
import torch
# x = torch.rand(5, 5)
# y = torch.rand(5, 5)
# y.add_(x)
# print(y)
#print(x[:, 1])
x = torch.randn(5, 5)
y = x.view(15)
z = x.view(-1,5)
print(x.size(), y.size(), z.size())