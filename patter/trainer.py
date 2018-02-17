import time
import torch
from torch.utils.data import DataLoader
from marshmallow.exceptions import ValidationError
from .config import TrainerConfiguration
from .data import BucketingSampler, audio_seq_collate_fn
from .util import AverageMeter


class Trainer(object):
    def __init__(self, train_config, tqdm=False):
        self.cfg = train_config['trainer']
        self.output = train_config['output']
        self.cuda = train_config['cuda']
        self.train_id = train_config['expt_id']
        self.tqdm=tqdm
        self.max_norm = self.cfg.get('max_norm', None)

    def train(self, model, corpus, eval=None):
        train_sampler = BucketingSampler(corpus, batch_size=self.cfg['batch_size'])
        train_loader = DataLoader(corpus, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                  pin_memory=True, batch_sampler=train_sampler)
        if eval is not None:
            eval_loader = DataLoader(eval, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                     pin_memory=True)
        else:
            eval_loader = None

        print("train:", self._train_config)
        print("\nmodel:", model)
        print("\ncorpus:", corpus)

    def train_epoch(self, train_loader, model, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        wers = AverageMeter()
        cers = AverageMeter()

        model.train()

        end = time.time()
        for i, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # create variables
            feat, target, feat_len, target_len = tuple(torch.autograd.Variable(i, requires_grad=False) for i in data)

            # compute output
            output, output_len = model(feat, feat_len)
            loss = model.loss(output, output_len, target, target_len)

            # decode measure accuracy and record loss

            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            if self.max_norm:
                torch.nn.utils.clip_grad_norm(model.parameters(), self.max_norm)
            optimizer.step()

            # measure time taken
            batch_time.update(time.time() - end)
            end = time.time()
        pass

    def validate(self, val_loader, model):
        batch_time = AverageMeter()
        losses = AverageMeter()
        wers = AverageMeter()
        cers = AverageMeter()
        pass

    @classmethod
    def load(cls, trainer_config, tqdm=False):
        try:
            cfg = TrainerConfiguration().load(trainer_config)
        except ValidationError as err:
            print(err.messages)
            raise err
        return cls(cfg.data, tqdm=tqdm)
