import torch.nn as nn


class SpeechModel(nn.Module):
    def __init__(self, cfg):
        super(SpeechModel, self).__init__()
        self._config = cfg
        self.input_cfg = cfg['input']

    def forward(self, _input, lengths):
        raise NotImplementedError

    def loss(self, x, y, x_length=None, y_length=None):
        raise NotImplementedError

    def init_weights(self):
        pass

    def get_output_offset_time_in_ms(self, offsets):
        raise NotImplementedError

    @property
    def is_cuda(self) -> bool:
        return next(self.parameters()).is_cuda

    @property
    def config(self) -> dict:
        return self._config

    def get_filter_images(self):
        return []

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, meta=None):
        # model_is_cuda = next(model.parameters()).is_cuda
        # model = model.module if model_is_cuda else model
        package = {
            'config': model.config,
            'state_dict': model.state_dict()
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

