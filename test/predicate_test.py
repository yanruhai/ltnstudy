import ltn
import torch
predicate_model = torch.nn.Sequential(
                             torch.nn.Linear(4, 2),
                             torch.nn.ELU(),
                             torch.nn.Linear(2, 1),
                             torch.nn.Sigmoid()
                       )
p = ltn.Predicate(model=predicate_model)
print(p)

p_f = ltn.Predicate(func=lambda x: torch.nn.Sigmoid()(torch.sum(x, dim=1)))#这里先初始化sigmoid函数指针，然后调用它的前向
print(p_f)
print("-----------------------")
class PredicateModel(torch.nn.Module):
    def __init__(self):
        super(PredicateModel, self).__init__()
        elu = torch.nn.ELU()
        sigmoid = torch.nn.Sigmoid()
        self.dense1 = torch.nn.Linear(4, 5)
        self.dense2 = torch.nn.Linear(5, 1)

        def forward(self, x, y):
            x = torch.cat([x, y], dim=1)
            x = self.elu(self.dense1(x))
            out = self.sigmoid(self.dense2(x))
            return out

predicate_model = PredicateModel()
b_p = ltn.Predicate(model=predicate_model)
print(b_p)

print("-----------------------")
b_p_f = ltn.Predicate(func=lambda x, y: torch.nn.Sigmoid()(
                                                 torch.sum(torch.cat([x, y], dim=1), dim=1)))
print(b_p_f)

print("-----------------------")
c = ltn.Constant(torch.tensor([0.5, 0.01, 0.34, 0.001]))
out = p_f(c)
print(type(out))
print(out)
print(out.value)
print(out.free_vars)
print(out.shape())

print("-----------------------")

v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
                                        [0.32, 0.043]]))
out = p_f(v)
print(out)
print("-----------------------")

x = ltn.Variable('x', torch.tensor([[0.4, 0.3],
                                    [0.32, 0.043]]))
print(x.shape())
y = ltn.Variable('y', torch.tensor([[0.4, 0.04, 0.23],
                                         [0.2, 0.04, 0.32],
                                         [0.06, 0.08, 0.3]]))
print(y.shape())
out = b_p_f(x, y)
print(out)
print(out.value)