from marshmallow.exceptions import ValidationError
from .config import TrainerConfiguration


class Trainer(object):
    def __init__(self, train_config):
        self._train_config = train_config
        pass

    def train(self, model, corpus):
        print("train:", self._train_config)
        print("\nmodel:", model)
        print("\ncorpus:", corpus)

    @classmethod
    def load(cls, trainer_config):
        try:
            cfg = TrainerConfiguration().load(trainer_config)
        except ValidationError as err:
            print(err.messages)
            raise err
        return cls(cfg.data)
