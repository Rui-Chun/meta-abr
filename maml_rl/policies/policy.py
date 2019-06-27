import torch
import torch.nn as nn

from collections import OrderedDict


def weight_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, loss, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        grads = torch.autograd.grad(loss, self.parameters(),
            create_graph=not first_order)
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        # opt = torch.optim.RMSprop(self.parameters(), lr=step_size, alpha=0.9)
        # self.opt.zero_grad()
        # loss.backward()
        # self.opt.step()
        # updated_params = OrderedDict(self.named_parameters())

        return updated_params
