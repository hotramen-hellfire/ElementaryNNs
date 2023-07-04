import torch

x=torch.randn(3, requires_grad=True)
print(x)

y=x*2
print(y)

z=y*y*8
print(z)
#z=z.mean()
v=torch.tensor([1,2,3], dtype=torch.float32)
z.backward(v)
print(x.grad)
