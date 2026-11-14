import torch

import ltn

Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

x = ltn.Variable('x', torch.randn((10, 2))) # 10 values in R²
y = ltn.Variable('y', torch.randn((5, 2))) # 5 values in R²

c1 = ltn.Constant(torch.tensor([0.5, 0.0]))
c2 = ltn.Constant(torch.tensor([4.0, 2.0]))

Eq = ltn.Predicate(func=lambda x, y: torch.exp(-torch.norm(x - y, dim=1))) # predicate measuring similarity

print(Not(Eq(c1, c2)).value)
print(f"Implies(Eq(c1, c2), Eq(c2, c1)).value={Implies(Eq(c1, c2), Eq(c2, c1)).value}")

print(f"And(Eq(x, c1), Eq(x, c2)).shape()={And(Eq(x, c1), Eq(x, c2)).shape()}")

print(Or(Eq(x, c1), Eq(x, y)).shape())

Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
x = ltn.Variable('x', torch.randn((10, 2))) # 10 values in R²
y = ltn.Variable('y', torch.randn((5, 2))) # 5 values in R²

Eq = ltn.Predicate(func=lambda x, y: torch.exp(-torch.norm(x - y, dim=1))) # predicate measuring similarity

Eq(x, y).shape()
print(Forall(x, Eq(x, y)).shape())

print(Exists([x, y], Eq(x, y)).value)

print(Forall(x, Exists(y, Eq(x, y))).value)


# The values are generated at random, for the sake of illustration.
# In a real scenario, they would come from a dataset.
samples = torch.randn((100, 2, 2)) # 100 R^{2x2} values
labels = torch.randint(0, 3, size=(100,)) # 100 labels (class 0/1/2) that correspond to each sample
onehot_labels = torch.nn.functional.one_hot(labels, num_classes=3)

x = ltn.Variable("x", samples)
l = ltn.Variable("l", onehot_labels)

class ModelC(torch.nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dense1 = torch.nn.Linear(4, 5)
        self.dense2 = torch.nn.Linear(5, 3)

    def forward(self, x, l):
        x = torch.flatten(x, start_dim=1)#start_dim默认值0，因此要处理下
        x = self.elu(self.dense1(x))
        x = self.softmax(self.dense2(x))
        return torch.sum(x * l, dim=1)

C = ltn.Predicate(ModelC())

print(C(x, l).shape()) # Computes the 100x100 combinations
ltn.diag(x, l) # sets the diag behavior for x and l
print(C(x, l).shape())# Computes the 100 zipped combinations

print(x.free_vars)
print(l.free_vars)
ltn.undiag(x, l) # resets the normal behavior
print(C(x, l).shape()) # Computes the 100x100 combinations

x, l = ltn.diag(x, l)
print(x.free_vars)
print(l.free_vars)
print(Forall([x, l], C(x, l)).value) # Aggregates only on the 100 "zipped" pairs.
                                    # Automatically calls `ltn.undiag` so the behavior of x/l is unchanged outside of this formula.
print(x.free_vars)
print(l.free_vars)

Eq = ltn.Predicate(func=lambda x, y: torch.exp(-torch.norm(x - y, dim=1))) # predicate measuring similarity

points = torch.rand((50, 2)) # 50 values in [0,1]^2
x = ltn.Variable("x", points)
y = ltn.Variable("y", points)
d = ltn.Variable("d", torch.tensor([.1,.2,.3,.4,.5,.6,.7,.8,.9]))

dist = lambda x, y: torch.unsqueeze(torch.norm(x.value - y.value, dim=1), 1) # function measuring euclidian distance
print(Exists(d,
      Forall([x, y],
            Eq(x, y),
            cond_vars=[x, y, d],
            cond_fn=lambda x, y, d: dist(x, y) < d.value
            )).value)

