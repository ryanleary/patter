import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class LookaheadConvolution(nn.Module):
    """
        Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
        input shape - sequence, batch, feature - TxNxH
        output shape - same as input
    """

    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(LookaheadConvolution, self).__init__()
        assert context > 0

        self.n_features = n_features
        self.weight = nn.Parameter(torch.Tensor(n_features, context + 1))
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialize this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input_):
        seq_len = input_.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input_.size()[1:])).type_as(input_.data)
        x = torch.cat((input_, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        x = [x[i:i + self.context + 1] for i in range(seq_len)]  # TxLxNxH - sequence, context, batch, feature
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(features={}, context={})'.format(self.n_features, self.context)
