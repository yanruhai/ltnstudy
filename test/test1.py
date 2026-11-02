import torch

x = torch.tensor([
    [1, 2, 3],
                  [4, 5, 6]
])
x=torch.unsqueeze(x,dim=1)
print(x)
x=torch.repeat_interleave(x, repeats=2, dim=1)
print(x)