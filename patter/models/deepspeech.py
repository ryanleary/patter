import math
import torch.nn as nn

from patter.models.model import SpeechModel
from .layer import NoiseRNN, LookaheadConvolution
from .activation import InferenceBatchSoftmax, Swish


activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "swish": Swish
}

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class DeepSpeechOptim(SpeechModel):
    def loss(self, x, y, x_length=None, y_length=None):
        pass

    def __init__(self, cfg):
        super(DeepSpeechOptim, self).__init__()
        self._config = cfg

        self.conv = self._get_cnn_layers(cfg['cnn'])

        rnn_input_size = self._get_rnn_input_size(cfg['input']['sample_rate'], cfg['input']['window_size'])
        self.rnns = NoiseRNN(input_size=rnn_input_size, hidden_size=cfg['rnn']['size'],
                             bidirectional=cfg['rnn']['bidirectional'], num_layers=cfg['rnn']['layers'],
                             rnn_type=supported_rnns[cfg['rnn']['rnn_type']],
                             weight_noise=cfg['rnn']['noise'])

        # generate the optional lookahead layer and fully-connected layer
        output = []
        if not cfg['rnn']['bidirectional']:
            output.append(LookaheadConvolution(cfg['rnn']['size'], context=cfg['ctx']['context']))
            output.append(activations[cfg['ctx']['activation']](*cfg['ctx']['activation_params']))
        output.append(nn.Linear(cfg['rnn']['size'], len(cfg['labels']['labels'])))

        self.output = nn.Sequential(*output)
        self.inference_softmax = InferenceBatchSoftmax()

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.

        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv:
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()

    @staticmethod
    def _get_cnn_layers(cfg):
        """
        Given the array of cnn configuration objects, create a sequential model consisting of Conv2d layers,
        optional batchnorm, and an activation function.
        :param cfg: array of CNN configuration objects
        :return: nn.Sequential of CNNs, BN, and Activations
        """
        cnns = []
        for x, cnn_cfg in enumerate(cfg):
            in_filters = cfg[x-1]['filters'] if x > 0 else 1
            cnn = nn.Conv2d(in_filters, cnn_cfg['filters'],
                            kernel_size=tuple(cnn_cfg['kernel']),
                            stride=tuple(cnn_cfg['stride']),
                            padding=tuple(cnn_cfg['padding']))
            cnns.append(cnn)
            if cnn_cfg['batch_norm']:
                cnns.append(nn.BatchNorm2d(cnn_cfg['filters']))
            cnns.append(activations[cnn_cfg['activation']](*cnn_cfg['activation_params']))
        return nn.Sequential(*cnns)

    def _get_rnn_input_size(self, sample_rate, window_size):
        """
        Calculate the size of tensor generated for a single timestep by the convolutional network
        :param sample_rate: number of samples per second
        :param window_size: size of windows as a fraction of a second
        :return: Size of hidden state
        """
        size = int(math.floor((sample_rate * window_size) / 2) + 1)
        channels = 0
        for mod in self.conv:
            if type(mod) == nn.modules.conv.Conv2d:
                size = math.floor(
                    (size + 2 * mod.padding[0] - mod.dilation[0] * (mod.kernel_size[0] - 1) - 1) / mod.stride[0] + 1)
                channels = mod.out_channels
        return size * channels

    def forward(self, x, lengths):
        """
        Perform a forward pass through the DeepSpeech model. Inputs are a batched spectrogram Variable and a Variable
        that indicates the sequence lengths of each example.

        The output (in inference mode) is a Variable containing posteriors over each character class at each timestep
        for each example in the minibatch.

        :param x: (batch_size, 1, stft_size, max_seq_len) Raw single-channel spectrogram input
        :param lengths: (batch,) Sequence_length for each sample in batch
        :return: FloatTensor(batch_size, max_seq_len, num_classes), IntTensor(batch_size)
        """
        x = self.conv(x)

        # collapse cnn channels into a feature vector per timestep
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        # convert padded matrix to PackedSequence, run rnn, and convert back
        output_lengths = self.get_seq_lens(lengths).data.tolist()
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, _ = self.rnns(x)
        x = nn.utils.rnn.pad_packed_sequence(x)

        # fully connected layer to output classes
        x = self.output(x)
        x = x.transpose(0, 1)

        # if training, return only logits (ctc loss calculates softmax), otherwise do softmax
        x = self.inference_softmax(x)
        return x, output_lengths
