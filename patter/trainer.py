class Trainer(object):
    def __init__(self, train_config):
        self._train_config = train_config
        pass

    def train(self, model_config, corpus_config):
        print("train:", self._train_config, model_config, corpus_config)

    @classmethod
    def load(cls, trainer_config):
        return cls(trainer_config)
