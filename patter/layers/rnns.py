import torch
import torch.nn as nn

from collections import OrderedDict
from .sequence import SequenceWise


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = x._replace(data=self.batch_norm(x.data))
        x, h = self.rnn(x)
        if self.bidirectional:
            x = x._replace(data=x.data[:, :self.hidden_size] + x.data[:, self.hidden_size:]) # sum bidirectional outputs
        return x


class DeepBatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, num_layers=1,
                 batch_norm=True, sum_directions=True, **kwargs):
        super(DeepBatchRNN, self).__init__()
        self._bidirectional = bidirectional
        rnns = []
        rnn = BatchRNN(input_size=input_size, hidden_size=hidden_size, rnn_type=rnn_type, bidirectional=bidirectional,
                       batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(num_layers - 1):
            rnn = BatchRNN(input_size=hidden_size, hidden_size=hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional, batch_norm=batch_norm)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.sum_directions = sum_directions

    def flatten_parameters(self):
        for x in range(len(self.rnns)):
            self.rnns[x].flatten_parameters()

    def forward(self, x, lengths):
        max_seq_length = x.size(0)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.data.squeeze(0).cpu().numpy())
        x = self.rnns(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=max_seq_length)
        return x, None


class NoiseRNN(nn.Module):
    """
    NoiseRNN wraps an arbitrary RNN class and adds weight noise to the weight parameters during training time.
    The noise is drawn from a Normal distribution and the mean/stdev of the distribution may be configured by passing
    a `(mean, stdev)` tuple to the `weight_noise` parameter of the NoiseRNN initialization.
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, num_layers=1,
                 weight_noise=None, sum_directions=True, **kwargs):
        super(NoiseRNN, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._noise = weight_noise
        self._sum_directions = sum_directions
        self.module = rnn_type(input_size=input_size, hidden_size=hidden_size,
                               bidirectional=bidirectional, bias=True, num_layers=num_layers)

        scratch_tensors = set([])
        for p, x in self.module.named_parameters():
            if not p.startswith("bias") and self._get_noise_buffer_name(x) not in scratch_tensors:
                self.register_buffer(self._get_noise_buffer_name(x), torch.zeros(x.shape).type_as(x.data))

    def flatten_parameters(self):
        self.module.flatten_parameters()

    def forward(self, x, lengths):
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.data.tolist())
        if self.training and self._noise is not None:
            # generate new set of random vectors
            for _, tensor in self._buffers.items():
                tensor.normal_(mean=self._noise['mean'], std=self._noise['std'])

            # add random vectors to weights
            for pn, pv in self.module.named_parameters():
                if not pn.startswith("bias"):
                    pv.data.add_(self._get_noise_buffer(pv))
        x, h = self.module(x)

        if self.training and self._noise is not None:
            # remove random vectors from weights
            for pn, pv in self.module.named_parameters():
                if not pn.startswith("bias"):
                    pv.data.sub_(self._get_noise_buffer(pv))

        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        # collapse fwd/bwd output if bidirectional rnn, otherwise do lookahead convolution
        if self._bidirectional and self._sum_directions:
            # (TxNxH*2) -> (TxNxH) by sum
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x, h

    @staticmethod
    def _get_noise_buffer_name(tensor):
        shape = tensor.shape
        out = str(shape[0])
        for x in range(1, len(shape)):
            out += "x" + str(shape[x])
        return "rnd_" + out

    def _get_noise_buffer(self, tensor):
        name = self.get_noise_buffer_name(tensor)
        return getattr(self, name)

    def __repr__(self):
        if self._noise is None:
            noise = "None"
        else:
            noise = "N({}, {})".format(self._noise['mean'], self._noise['std'])
        return self.__class__.__name__ + '(rnn={}, noise={}, sum_directions={})'.format(
            self.module, noise, self._sum_directions)
