import torch.nn as nn


class SpeechModel(nn.Module):
    def forward(self, _input, lengths):
        raise NotImplementedError

    def loss(self, x, y, x_length=None, y_length=None):
        raise NotImplementedError

    @property
    def is_cuda(self) -> bool:
        return self.parameters().next().is_cuda

    @property
    def config(self) -> dict:
        return self._config
