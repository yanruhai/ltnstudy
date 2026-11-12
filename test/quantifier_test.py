import ltn
import torch

p = ltn.Predicate(func=lambda a, b: torch.nn.Sigmoid()(
                                         torch.sum(torch.cat([a, b], dim=1),
                                       dim=1)))
'''ax=ltn.Variable('ax',torch.rand((4,5)))
ay=ltn.Variable('ay',torch.rand((6,7)))
t=p(ax,ay)
print(t.shape)#结果是4,6的张量'''
x = ltn.Variable('x', torch.tensor([[0.3, 1.3],
                                         [0.3, 0.3]]))
y = ltn.Variable('y', torch.tensor([[2.3, 0.3, 0.4],
                                         [1.2, 3.4, 1.3],
                                         [2.3, 1.4, 1.4]]))
out = p(x, y)
print(out)
#print('out.shape:'+str(out.shape()))

Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(), quantifier='f')
print(Forall)
print(f'x.shape={x.value.shape},y.shape={y.value.shape},p(x,y).shape={out.value.shape}')
out = Forall(x, p(x, y))
print(f"after froall-x shape={out.value.shape}")
print(out)
print(out.value)
print(out.free_vars)
print(out.shape())

