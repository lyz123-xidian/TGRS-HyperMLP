import torch

a = torch.randn((12,121,128))
b = torch.randn((12,255,128))

a1 = a.mean(dim=1)
b1 = b.mean(dim=1)

c = torch.randn((12,128))

c1 = c.repeat(121,1,1)
d = torch.mul(a,c)
e = torch.mul(b,c)

print(d.shape())