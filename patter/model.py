from marshmallow.exceptions import ValidationError
from .models.deepspeech import DeepSpeechOptim
from .models.model import SpeechModelConfiguration


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
