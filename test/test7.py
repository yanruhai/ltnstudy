import torch

# 假设一个一维布尔张量，长度为 5
x = torch.tensor([True, False, True, False, True])
print("原数据形状:", x.shape)  # torch.Size([5])

# 1. logical_not 的结果
res_not = torch.logical_not(x)
print("logical_not 结果形状:", res_not.shape)  # torch.Size([5]) —— 维度保持不变

# 2. nonzero 的结果
res_nonzero = torch.nonzero(x)
print("nonzero 结果形状:", res_nonzero.shape)       # torch.Size([3, 1]) —— 变成 2D 索引张量 (3个True，输入是1维)
print(res_nonzero)

import torch

data = torch.randn(50, 2)
# 模拟 torch.nonzero() 返回的形状为 (3, 1) 的索引
indices = torch.tensor([[0], [2], [4]])

result = data[indices]
print(result.shape)