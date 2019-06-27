import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)

A_DIM = 6
# ENTROPY_WEIGHT = 0.001  # too high, making entropy too small
ENTROPY_WEIGHT = 0.0
ENTROPY_EPS = 1e-8

class ActorNet(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a
    `Categorical` distribution output. This policy network can be used on tasks
    with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
    """
    def __init__(self, input_size, output_size, learning_rate=0.0001):
        super(ActorNet, self).__init__(
            input_size=input_size, output_size=output_size)

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
        self.add_module('layer2-fc', nn.Linear(128, A_DIM))

        self.apply(weight_init)

        self.learning_rate = learning_rate
        self.opt = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

        self.paramsFlag = OrderedDict(self.named_parameters())

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
        split1 = F.linear(output[:, 1, -1].view(state_size, 1),
                          weight=params['layer0-fc1.weight'],
                          bias=params['layer0-fc1.bias'])
        split2 = F.conv1d(output[:, 2, :].view(state_size, 1, -1),
                          weight=params['layer0-cv0.weight'],
                          bias=params['layer0-cv0.bias'])
        split2 = split2.view(state_size, -1)
        split3 = F.conv1d(output[:, 3, :].view(state_size, 1, -1),
                          weight=params['layer0-cv1.weight'],
                          bias=params['layer0-cv1.bias'])
        split3 = split3.view(state_size, -1)
        split4 = F.conv1d(output[:, 4, :A_DIM].view(state_size, 1, -1),
                          weight=params['layer0-cv2.weight'],
                          bias=params['layer0-cv2.bias'])
        split4 = split4.view(state_size, -1)
        split5 = F.linear(output[:, 5, -1].view(state_size, 1),
                          weight=params['layer0-fc2.weight'],
                          bias=params['layer0-fc2.bias'])

        merge = torch.cat([split0, split1, split2, split3, split4, split5], 1)
        merge = F.relu(merge)

        output = F.linear(merge,
                          weight=params['layer1-fc.weight'],
                          bias=params['layer1-fc.bias'])
        output = F.relu(output)

        output = F.linear(output,
            weight=params['layer2-fc.weight'],
            bias=params['layer2-fc.bias'])
        probs = F.softmax(output, dim=1)

        probs = probs.reshape(state_size0, state_size1, A_DIM)
        return Categorical(probs)

    def fit(self, episodes, baseline,  learningrate=0.0001, isnew=False):
        if isnew:
            self.opt = torch.optim.RMSprop(self.parameters(), lr=learningrate, alpha=0.9)

        batchnum = episodes.observations.size()[1]
        values = baseline(episodes.observations)
        advantages = episodes.gae(values, tau=1)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        loss_list = []
        for i in range(batchnum):
            pi = self.forward(episodes.observations[:, i:i+1, :, :])
            log_probs = pi.log_prob(episodes.actions[:, i:i+1])
            if log_probs.dim() > 2:
                log_probs = torch.sum(log_probs, dim=2)

            loss = -weighted_mean(log_probs * advantages[:, i:i+1], dim=0,
                weights=episodes.mask) - ENTROPY_WEIGHT * torch.sum(pi.logits * (pi.probs + ENTROPY_EPS))

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_list.append(float(loss))

        loss_list = np.array(loss_list)
        return loss_list.mean()





