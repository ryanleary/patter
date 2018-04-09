import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from collections import OrderedDict, defaultdict
from patter.models.model import SpeechModel

from .activation import InferenceBatchSoftmax, Swish

try:
    from warpctc_pytorch import CTCLoss
except ImportError:
    print("WARN: CTCLoss not imported. Use only for inference.")
    CTCLoss = lambda x, y: 0

activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "swish": Swish
}


class GatedConvLayer(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, padding=0, stride=1, dropout=0.0, bias=True):
        self.cnn = nn.Conv1D(in_channels=in_size, out_channes=out_size, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(F.glu(self.cnn(x), dim=1))
        return x


class TDNNOptim(SpeechModel):
    def __init__(self, cfg):
        super(TDNNOptim, self).__init__(cfg)
        self.input_cfg = cfg['input']
        self.loss_func = None

        # Add a `\u00a0` (no break space) label as a "BLANK" symbol for CTC
        self.labels = ['\u00a0'] + cfg['labels']['labels']

        self.conv = nn.Sequential(
            GatedConvLayer(in_size=161, out_size=400, kernel_size=13, stride=1, dropout=0.2),
            GatedConvLayer(in_size=200, out_size=440, kernel_size=14, stride=1, dropout=0.214),
            GatedConvLayer(in_size=220, out_size=484, kernel_size=15, stride=1, dropout=0.22898),
            GatedConvLayer(in_size=242, out_size=532, kernel_size=16, stride=1, dropout=0.2450086),
            GatedConvLayer(in_size=266, out_size=584, kernel_size=17, stride=1, dropout=0.262159202),
            GatedConvLayer(in_size=292, out_size=642, kernel_size=18, stride=1, dropout=0.28051034614),
            GatedConvLayer(in_size=321, out_size=706, kernel_size=19, stride=1, dropout=0.30014607037),
            GatedConvLayer(in_size=353, out_size=776, kernel_size=20, stride=1, dropout=0.321156295296),
            GatedConvLayer(in_size=388, out_size=852, kernel_size=21, stride=1, dropout=0.343637235966),
            GatedConvLayer(in_size=426, out_size=936, kernel_size=22, stride=1, dropout=0.367691842484),
            GatedConvLayer(in_size=468, out_size=1028, kernel_size=23, stride=1, dropout=0.393430271458),
            GatedConvLayer(in_size=514, out_size=1130, kernel_size=24, stride=1, dropout=0.42097039046),
            GatedConvLayer(in_size=565, out_size=1242, kernel_size=25, stride=1, dropout=0.450438317792),
            GatedConvLayer(in_size=621, out_size=1366, kernel_size=26, stride=1, dropout=0.481969000038),
            GatedConvLayer(in_size=683, out_size=1502, kernel_size=27, stride=1, dropout=0.51570683004),
            GatedConvLayer(in_size=751, out_size=1652, kernel_size=28, stride=1, dropout=0.551806308143),
            GatedConvLayer(in_size=826, out_size=1816, kernel_size=29, stride=1, dropout=0.590432749713),
            GatedConvLayer(in_size=908, out_size=1816, kernel_size=30, stride=1, dropout=0.590432749713),
            nn.Conv1d(in_channels=908, out_channels=len(self.labels), kernel_size=1)
        )

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
            self.loss_func = CTCLoss()
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
        return self.loss_func(x, y, x_length, y_length)

    def init_weights(self):
        """
        Initialize weights with sensible defaults for the various layer types
        :return:
        """
        # ih = ((name, param.data) for name, param in self.named_parameters() if 'weight_ih' in name)
        # hh = ((name, param.data) for name, param in self.named_parameters() if 'weight_hh' in name)
        b = ((name, param.data) for name, param in self.named_parameters() if 'bias' in name)
        w = ((name, param.data) for name, param in self.named_parameters() if 'weight' in name and 'rnn' not in name and "batch_norm" not in name)
        bn_w = ((name, param.data) for name, param in self.named_parameters() if 'batch_norm' in name and 'weight' in name)

        # for t in ih:
        #     nn.init.xavier_uniform(t[1])
        for t in w:
            nn.init.xavier_uniform(t[1])
        # for t in hh:
        #     nn.init.orthogonal(t[1])
        for t in b:
            nn.init.constant(t[1], 0)
        for t in bn_w:
            nn.init.constant(t[1], 1)

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
            cnns.append(("{}.cnn".format(x), cnn),)
            if cnn_cfg['batch_norm']:
                cnns.append(("{}.batch_norm".format(x), nn.BatchNorm2d(cnn_cfg['filters'])))
            cnns.append(("{}.act".format(x), activations[cnn_cfg['activation']](*cnn_cfg['activation_params'])),)
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
        x = self.conv(x.squeeze())
        print(x.shape)

        # if training, return only logits (ctc loss calculates softmax), otherwise do softmax
        x = self.inference_softmax(x, dim=2)
        del lengths
        return x

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
