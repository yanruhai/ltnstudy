import torch

a=torch.tensor([[1,2],[3,4]])#2,2
b=torch.tensor([[5,6],[7,8]])

d=torch.rand((2,2))

c=torch.stack((a,b,d,a),dim=0)
print(c)
print(c.shape)