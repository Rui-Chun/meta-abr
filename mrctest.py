import torch
from torch.distributions import Categorical
import numpy as np
#
# a = torch.tensor([[[1, 2], [3, 4]],[[1,2],[3,4]]])
# print(a, a.size())
#
# a = a.reshape(4, 2)
#
# print(a, a.size())
#
# a = a.reshape(2,2,2)
#
# print(a, a.size())
#
#
# probs = torch.tensor([0.1, 0.1, 0.3, 0.5])
# # Note that this is equivalent to what used to be called multinomial
# m = Categorical(probs)
# action = m.sample()
#
# loss = -m.log_prob(action) * 1
# loss.backward()

a=np.array([[1,2],[3,4]])
b=np.array([[2,3],[4,4]])
loss_fn = torch.nn.MSELoss()
input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))
loss = loss_fn(input.float(), target.float())
print(loss)
