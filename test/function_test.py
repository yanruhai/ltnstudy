import ltn
import torch
function_model = torch.nn.Sequential(
                         torch.nn.Linear(4, 3),
                         torch.nn.ELU(),
                         torch.nn.Linear(3, 2))
f = ltn.Function(model=function_model)
print(f)

f_f = ltn.Function(func=lambda x: torch.repeat_interleave(
                                                torch.sum(x, dim=1, keepdim=True), 2, dim=1))
print(f_f)

print("-------------------")

class FunctionModel(torch.nn.Module):
    def __init__(self):
        super(FunctionModel, self).__init__()
        elu = torch.nn.ELU()
        self.dense1 = torch.nn.Linear(4, 5)
        dense2 = torch.nn.Linear(5, 2)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.elu(self.dense1(x))
        out = self.dense2(x)
        return out

function_model = FunctionModel()
b_f = ltn.Function(model=function_model)
print(b_f)

print("-------------------")
b_f_f = ltn.Function(func=lambda x, y:
    torch.repeat_interleave(
    torch.sum(torch.cat([x, y], dim=1), dim=1, keepdim=True), 2,
    dim=1))
print(b_f_f)

print("-------------------")
c = ltn.Constant(torch.tensor([0.5, 0.01, 0.34, 0.001]))
out = f_f(c)
print(type(out))
print(out)

print(out.value)
print(out.free_vars)
print(out.shape())

print("-------------------")
v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
                                           [0.32, 0.043]]))
out = f_f(v)
print(out)

print("-------------------")
v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
                                         [0.32, 0.043]]))
c = ltn.Constant(torch.tensor([0.4, 0.04, 0.23, 0.43]))
out = b_f_f(v, c)
print(out)

print("-------------------")
x = ltn.Variable('x', torch.tensor([[0.4, 0.3],
                                             [0.32, 0.043]]))
y = ltn.Variable('y', torch.tensor([[0.4, 0.04, 0.23],
                                         [0.2, 0.04, 0.32],
                                         [0.06, 0.08, 0.3]]))
out = b_f_f(x, y)
print(out)
print(out.value)

print(out.shape())

print(out.value[1, 2])