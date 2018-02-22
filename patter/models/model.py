import torch.nn as nn


class SpeechModel(nn.Module):
    def __init__(self):
        super(SpeechModel, self).__init__()

    def forward(self, _input, lengths):
        raise NotImplementedError

    def loss(self, x, y, x_length=None, y_length=None):
        raise NotImplementedError

    @property
    def is_cuda(self) -> bool:
        return next(self.parameters()).is_cuda

    @property
    def config(self) -> dict:
        return self._config
