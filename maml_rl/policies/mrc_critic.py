import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

A_DIM = 6

class CriticNet(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a
    `Categorical` distribution output. This policy network can be used on tasks
    with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
    """

    def __init__(self, input_size, output_size, learning_rate=0.001):
        super(CriticNet, self).__init__(input_size=input_size, output_size=output_size)

        # for i in range(1, self.num_layers + 1):
        #     self.add_module('layer{0}'.format(i),
        #         nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.add_module('layer0-fc0', nn.Linear(1, 128))
        self.add_module('layer0-fc1', nn.Linear(1, 128))
        self.add_module('layer0-cv0', nn.Conv1d(1, 128, 4))
        self.add_module('layer0-cv1', nn.Conv1d(1, 128, 4))
        self.add_module('layer0-cv2', nn.Conv1d(1, 128, 4))
        self.add_module('layer0-fc2', nn.Linear(1, 128))

        self.add_module('layer1-fc', nn.Linear(2048, 128))  # 输入维度待定~~~~~~~~~~~~~~~
        self.add_module('layer2-fc', nn.Linear(128, 1))

        self.apply(weight_init)

        self.learning_rate = learning_rate
        self.opt = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        # output = torch.tensor(input)
        output = torch.tensor(input)
        output = output.float()
        state_size0, state_size1, state_size2, state_size3 = output.size()

        state_size = state_size0 * state_size1
        output = output.reshape(state_size, state_size2, state_size3)

        # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
        # print(output[:, 0, -1].size())
        split0 = F.linear(output[:, 0, -1].view(state_size, 1),
                          weight=params['layer0-fc0.weight'],
                          bias=params['layer0-fc0.bias'])
        split0 = F.relu(split0)

        split1 = F.linear(output[:, 1, -1].view(state_size, 1),
                          weight=params['layer0-fc1.weight'],
                          bias=params['layer0-fc1.bias'])
        split1 = F.relu(split1)

        split2 = F.conv1d(output[:, 2, :].view(state_size, 1, -1),
                          weight=params['layer0-cv0.weight'],
                          bias=params['layer0-cv0.bias'])
        split2 = split2.view(state_size, -1)
        split2 = F.relu(split2)

        split3 = F.conv1d(output[:, 3, :].view(state_size, 1, -1),
                          weight=params['layer0-cv1.weight'],
                          bias=params['layer0-cv1.bias'])
        split3 = split3.view(state_size, -1)
        split3 = F.relu(split3)

        split4 = F.conv1d(output[:, 4, :A_DIM].view(state_size, 1, -1),
                          weight=params['layer0-cv2.weight'],
                          bias=params['layer0-cv2.bias'])
        split4 = split4.view(state_size, -1)
        split4 = F.relu(split4)

        split5 = F.linear(output[:, 5, -1].view(state_size, 1),
                          weight=params['layer0-fc2.weight'],
                          bias=params['layer0-fc2.bias'])
        split5 = F.relu(split5)

        merge = torch.cat([split0, split1, split2, split3, split4, split5], 1)
        merge = F.relu(merge)

        output = F.linear(merge,
                          weight=params['layer1-fc.weight'],
                          bias=params['layer1-fc.bias'])
        output = F.relu(output)

        output = F.linear(output,
            weight=params['layer2-fc.weight'],
            bias=params['layer2-fc.bias'])

        output = output.reshape(state_size0, state_size1, 1)

        return output

    def fit(self, episodes, learning_rate=0.001, isnew=False):
        if isnew:  # start a new optimizer for meta-training
            self.opt = torch.optim.RMSprop(self.parameters(), lr=learning_rate, alpha=0.9)

        returns = episodes.returns
        loss_func = torch.nn.MSELoss()

        loss_list = []
        for i in range(returns.size()[1]):
            values = self.forward(episodes.observations[:, i:i+1, :, :]).view(-1)
            loss = loss_func(values, returns[:, i])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_list.append(float(loss))

        loss_list = np.array(loss_list)
        return loss_list.mean()

