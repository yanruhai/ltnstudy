import ltn
import torch

p = ltn.Predicate(func=lambda a: torch.nn.Sigmoid()(
                                     torch.sum(a, dim=1)
                                   ))
q = ltn.Predicate(func=lambda a, b: torch.nn.Sigmoid()(
                                    torch.sum(torch.cat([a, b], dim=1),
                                 dim=1)))
x = ltn.Variable('x', torch.tensor([[0.3, 0.5],
                                    [0.04, 0.43]]))
y = ltn.Variable('y', torch.tensor([[0.5, 0.23],
                                  [4.3, 9.3],
                                   [4.3, 0.32]]))
z = ltn.Variable('z', torch.tensor([[0.3, 0.4, 0.43],
                                        [0.4, 4.3, 5.1],
                                        [1.3, 4.3, 2.3],
                                        [0.4, 0.2, 1.2]]))
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
print(And)
out = And(p(x), q(y, z))
print(out)
print(out.shape())