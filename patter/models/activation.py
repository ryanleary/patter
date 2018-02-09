import torch
from torch.nn import Module, functional as F


class Swish(Module):
    """Implementation of Swish: a Self-Gated Activation Function
        Swish activation is simply f(x)=x⋅sigmoid(x)
        Paper: https://arxiv.org/abs/1710.05941
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = nn.Swish()
        >>> x = autograd.Variable(torch.randn(2))
        >>> print(x)
        >>> print(m(x))
    """

    def forward(self, input_):
        return input_ * torch.sigmoid(input_)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class InferenceBatchSoftmax(Module):
    """ Implementation of the Softmax activation, that is only
        run during inference. During training, the input is
        returned unchanged.
    """
    def forward(self, input_, dim=-1):
        if not self.training:
            return F.softmax(input_, dim=dim)
        else:
            return input_

    def __repr__(self):
        return self.__class__.__name__ + '()'
