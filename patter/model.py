import torch
from marshmallow.exceptions import ValidationError
from .models.deepspeech import DeepSpeechOptim
from .config import SpeechModelConfiguration


class ModelFactory(object):
    _models = {
        "DeepSpeechOptim": DeepSpeechOptim
    }

    @classmethod
    def create(cls, cfg, model_type=None):
        if model_type is None:
            model_type = cfg['model']
        klass = cls._models[model_type]
        try:
            cfg = SpeechModelConfiguration().load(cfg)
        except ValidationError as err:
            print(err.messages)
            raise err
        return klass(cfg.data)

    @classmethod
    def load(cls, path, include_package=False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        cfg = package['config']

        model = cls.create(cfg)
        model.load_state_dict(package['state_dict'])
        model.flatten_parameters()

        if include_package:
            del package['state_dict']
            del package['optim_dict']
            return model, package

        return model
