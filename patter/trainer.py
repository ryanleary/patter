from torch.utils.data import DataLoader
from marshmallow.exceptions import ValidationError
from .config import TrainerConfiguration
from .data import BucketingSampler, audio_seq_collate_fn


class Trainer(object):
    def __init__(self, train_config, tqdm=False):
        self.cfg = train_config['trainer']
        self.output = train_config['output']
        self.cuda = train_config['cuda']
        self.train_id = train_config['expt_id']
        self.tqdm=tqdm

    def train(self, model, corpus, eval=None):
        train_sampler = BucketingSampler(corpus, batch_size=self.cfg['batch_size'])
        train_loader = DataLoader(corpus, batch_sampler=train_sampler, num_workers=self.cfg['num_workers'],
                                  collate_fn=audio_seq_collate_fn)
        if eval is not None:
            eval_loader = DataLoader(eval, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn)
        else:
            eval_loader = None

        print("train:", self._train_config)
        print("\nmodel:", model)
        print("\ncorpus:", corpus)

    @classmethod
    def load(cls, trainer_config, tqdm=False):
        try:
            cfg = TrainerConfiguration().load(trainer_config)
        except ValidationError as err:
            print(err.messages)
            raise err
        return cls(cfg.data, tqdm=tqdm)
