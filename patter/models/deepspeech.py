import math
import torch.nn as nn

from . import SpeechModel
from .layer import NoiseRNN, LookaheadConvolution
from .activation import InferenceBatchSoftmax


class DeepSpeechOptim(SpeechModel):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, num_layers=5, audio_conf=None,
                 bidirectional=True, context=20):
        super(DeepSpeechOptim, self).__init__()

        # model metadata needed for serialization/deserialization
        if audio_conf is None:
            audio_conf = {}
        self._version = '0.1.0'
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = num_layers
        self._rnn_type = rnn_type
        self._audio_conf = audio_conf or {}
        self._labels = labels
        self._bidirectional = bidirectional

        sample_rate = self._audio_conf.get("sample_rate", 16000)
        window_size = self._audio_conf.get("window_size", 0.02)

        self._activation = nn.Hardtanh(0, 20)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
            nn.BatchNorm2d(32),
            self._activation,
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
            nn.BatchNorm2d(32),
            self._activation
        )

        self.rnns = NoiseRNN(input_size=self.get_rnn_input_size(sample_rate, window_size), hidden_size=rnn_hidden_size,
                             bidirectional=bidirectional, num_layers=num_layers, rnn_type=rnn_type)

        if bidirectional:
            self.lookahead = None
        else:
            self.lookahead = nn.Sequential(LookaheadConvolution(rnn_hidden_size, context=context), self._activation)

        self.fc = nn.Linear(rnn_hidden_size, len(self._labels))
        self.inference_softmax = InferenceBatchSoftmax()

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.

        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for mod in self.conv:
            if type(mod) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * mod.padding[1] - mod.dilation[1] * (mod.kernel_size[1] - 1) - 1) / mod.stride[1] + 1)
        return seq_len.int()

    def get_rnn_input_size(self, sample_rate, window_size):
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
        print(x.shape)
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

        # collapse fwd/bwd output if bidirectional rnn, otherwise do lookahead convolution
        if self._bidirectional:
            # (TxNxH*2) -> (TxNxH) by sum
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        else:
            # do a lookahead convolution
            x = self.lookahead(x)

        # fully connected layer to output classes
        x = self.fc(x)
        x = x.transpose(0, 1)

        # if training, return only logits (ctc loss calculates softmax), otherwise do softmax
        x = self.inference_softmax(x)
        return x, output_lengths
