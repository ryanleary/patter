import math
import torch.nn as nn
import torchvision.utils as vutils

from collections import OrderedDict
from patter.models.model import SpeechModel
from patter.layers import NoiseRNN, DeepBatchRNN, LookaheadConvolution, SequenceWise
from .activation import InferenceBatchSoftmax, Swish

try:
    from warpctc import CTCLoss
except ImportError:
    print("WARN: CTCLoss not imported. Use only for inference.")
    CTCLoss = lambda x, y: 0

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
    def __init__(self, cfg):
        super(DeepSpeechOptim, self).__init__(cfg)
        self.loss_func = None

        # Add a `\u00a0` (no break space) label as a "BLANK" symbol for CTC
        self.labels = ['\u00a0'] + cfg['labels']['labels']

        # create the convolutional input layer(s)
        self.conv = self._get_cnn_layers(cfg['cnn'])

        # create the RNN(s)
        rnn_input_size = self._get_rnn_input_size(cfg['input']['sample_rate'], cfg['input']['window_size'])
        rnn_class = DeepBatchRNN if cfg['rnn']['batch_norm'] else NoiseRNN
        self.rnn = rnn_class(input_size=rnn_input_size, hidden_size=cfg['rnn']['size'],
                             bidirectional=cfg['rnn']['bidirectional'], num_layers=cfg['rnn']['layers'],
                             rnn_type=supported_rnns[cfg['rnn']['rnn_type']],
                             weight_noise=cfg['rnn']['noise'], batch_norm=cfg['rnn']['batch_norm'])

        # generate the optional lookahead layer and fully-connected layer
        output = []
        if not cfg['rnn']['bidirectional']:
            output.append(LookaheadConvolution(cfg['rnn']['size'], context=cfg['ctx']['context']))
            output.append(activations[cfg['ctx']['activation']](*cfg['ctx']['activation_params']))
        if cfg['rnn']['batch_norm']:
            output.append(SequenceWise(nn.BatchNorm1d(cfg['rnn']['size'])))
        output.append(nn.Linear(cfg['rnn']['size'], len(self.labels), bias=False))
        self.output = nn.Sequential(*output)

        # and output activation (softmax) ONLY at inference time (CTC applies softmax during training)
        self.inference_softmax = InferenceBatchSoftmax()
        self.init_weights()

    def train(self, mode=True):
        """
        Enter (or exit) training mode. Initializes loss function if necessary
        :param mode: if True, set model up for training
        :return:
        """
        if mode and self.loss_func is None:
            self.loss_func = CTCLoss(size_average=False)
        super().train(mode=mode)

    def loss(self, x, y, x_length=None, y_length=None):
        """
        Compute CTC loss for the given inputs
        :param x: predicted values
        :param y: reference values
        :param x_length: length of prediction
        :param y_length: length of references
        :return:
        """
        if self.loss_func is None:
            self.train()
        return self.loss_func(x, x_length, y, y_length)

    def init_weights(self):
        """
        Initialize weights with sensible defaults for the various layer types
        :return:
        """
        ih = ((name, param.data) for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = ((name, param.data) for name, param in self.named_parameters() if 'weight_hh' in name)
        b = ((name, param.data) for name, param in self.named_parameters() if 'bias' in name)
        w = ((name, param.data) for name, param in self.named_parameters() if 'weight' in name and 'rnn' not in name and "batch_norm" not in name and param.dim() > 1)

        for t in ih:
            nn.init.xavier_uniform_(t[1])
        for t in w:
            nn.init.xavier_uniform_(t[1])
        for t in hh:
            nn.init.orthogonal_(t[1])
        for t in b:
            nn.init.constant_(t[1], 0)

    def flatten_parameters(self):
        """
        Call flatten_parameters on underlying RNN(s)
        :return:
        """
        self.rnn.flatten_parameters()

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.

        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length.squeeze(0)
        for m in self.conv:
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int().unsqueeze(0)
    #
    # def get_output_offset_time_in_ms(self, offsets):
    #     seq_len = 0
    #     for m in self.conv:
    #         if type(m) == nn.modules.conv.Conv2d:
    #             seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
    #     offsets = (1/seq_len) * offsets *

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
            in_filters = cfg[x - 1]['filters'] if x > 0 else 1
            cnn = nn.Conv2d(in_filters, cnn_cfg['filters'],
                            kernel_size=tuple(cnn_cfg['kernel']),
                            stride=tuple(cnn_cfg['stride']),
                            padding=tuple(cnn_cfg['padding']))
            cnns.append(("{}-cnn".format(x), cnn),)
            if cnn_cfg['batch_norm']:
                cnns.append(("{}-bn".format(x), nn.BatchNorm2d(cnn_cfg['filters'])))
            cnns.append(("{}-act".format(x), activations[cnn_cfg['activation']](*cnn_cfg['activation_params'])),)
        return nn.Sequential(OrderedDict(cnns))

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

        :param x: (1, batch_size, stft_size, max_seq_len) Raw single-channel spectrogram input
        :param lengths: (batch,) Sequence_length for each sample in batch
        :return: FloatTensor(max_seq_len, batch_size, num_classes), IntTensor(batch_size)
        """

        # transpose to be of shape (batch_size, num_channels [1], height, width) and do CNN feature extraction
        x = self.conv(x.transpose(0,1))

        # collapse cnn channels into a feature vector per timestep
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        # calculate number of timesteps and run through RNN
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.rnn(x, output_lengths)

        # fully connected layer to output classes
        x = self.output(x)

        # if training, return only logits (ctc loss calculates softmax), otherwise do softmax
        x = self.inference_softmax(x, dim=2)
        del lengths, sizes
        return x, output_lengths

    def get_filter_images(self):
        """
        Generate a grid of images representing the convolution layer weights
        :return: list of images
        """
        images = []
        x = 0
        for mod in self.conv:
            if type(mod) == nn.modules.conv.Conv2d:
                orig_shape = mod.weight.data.shape
                weights = mod.weight.data.view(
                    [orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[3]]).unsqueeze(1)
                rows = 2 ** math.ceil(math.sqrt(math.sqrt(weights.shape[0])))
                images.append(("CNN.{}".format(x),
                               vutils.make_grid(weights, nrow=rows, padding=1, normalize=True, scale_each=True)))
            x += 1
        return images
