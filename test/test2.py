import torch

x = torch.tensor([
    [1, 2, 3],
                  [4, 5, 6]
])
t=torch.sum(x,dim=0)
t=torch.unsqueeze(t,dim=0)
x=torch.cat([x,t],dim=0)
print(x)
