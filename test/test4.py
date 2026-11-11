import torch

x = torch.randn(2, 3, 5)
print(x.size())
x=torch.permute(x, (1, 0))
print(x.size())
