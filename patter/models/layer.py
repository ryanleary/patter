import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class NoiseRNN(nn.Module):
    """
    NoiseRNN wraps an arbitrary RNN class and adds weight noise to the weight parameters during training time.
    The noise is drawn from a Normal distribution and the mean/stdev of the distribution may be configured by passing
    a `(mean, stdev)` tuple to the `weight_noise` parameter of the NoiseRNN initialization.
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, num_layers=1, weight_noise=None):
        super(NoiseRNN, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._num_directions = 2 if bidirectional else 1
        self._weight_noise = weight_noise
        self.module = rnn_type(input_size=input_size, hidden_size=hidden_size,
                               bidirectional=bidirectional, bias=True, num_layers=num_layers)

        scratch_tensors = set([])
        for p, x in self.module.named_parameters():
            if not p.startswith("bias") and self.get_noise_buffer_name(x) not in scratch_tensors:
                self.register_buffer(self.get_noise_buffer_name(x), torch.zeros(x.shape).type_as(x.data))

    def flatten_parameters(self):
        self.module.flatten_parameters()

    def forward(self, x):
        if self.training and self._weight_noise is not None:
            for pn, pv in self.module.named_parameters():
                if not pn.startswith("bias"):
                    pv.data.add_(self.get_noise_buffer(pv).normal_(mean=self._weight_noise[0], std=self._weight_noise[1]))
        x, h = self.module(x)
        return x, h

    @staticmethod
    def get_noise_buffer_name(tensor):
        shape = tensor.shape
        out = str(shape[0])
        for x in range(1, len(shape)):
            out += "x" + str(shape[x])
        return "rnd_" + out

    def get_noise_buffer(self, tensor):
        name = self.get_noise_buffer_name(tensor)
        return getattr(self, name)


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
        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input_.size()[1:])).type_as(input_.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        x = [x[i:i + self.context + 1] for i in range(seq_len)]  # TxLxNxH - sequence, context, batch, feature
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context

        x = torch.mul(x, self.weight).sum(dim=3)
        return x
