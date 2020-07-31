import torch
import torch.nn as nn

# a = torch.rand(2,3, 4)
# print(a)
# z = a.reshape(2,12)
# print(z)


# a = torch.Tensor(2, 4)
# print (a)
# b = a.view_as(torch.Tensor(4, 2))
# print (b)

# a = torch.rand(4)
# print(a)
# z=a.data.max(1,keepdim=True)
# print(z)
b= torch.tensor([[[1,2],[3,4]],[[5,5],[7,8]]],dtype=float)
print(b.shape)
m0 = nn.Softmax(dim=2)
x = m0(b)
print(x)