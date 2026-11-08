import ltn
import torch


p = ltn.Predicate(func=lambda a, b: torch.nn.Sigmoid()(
                                         torch.sum(torch.cat([a, b], dim=1), dim=1)                                 ))
x = ltn.Variable('x', torch.tensor([[0.3, 0.56, 0.43], [0.3, 0.5, 0.04]]))
y = ltn.Variable('y', torch.tensor([[0.4, 0.004], [0.3, 0.32]]))
out = p(x, y)
print(out.value)

print(out.free_vars)
print(out.shape())
print('--------------------')
x, y = ltn.diag(x, y)
out = p(x, y)
print(out.value)

print(out.free_vars)

print(out.shape())