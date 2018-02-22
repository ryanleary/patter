import math
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
        # set up data loaders
        train_sampler = BucketingSampler(corpus, batch_size=self.cfg['batch_size'])
        train_loader = DataLoader(corpus, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                  pin_memory=True, batch_sampler=train_sampler)
        if eval is not None:
            eval_loader = DataLoader(eval, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                     pin_memory=True)
        else:
            eval_loader = None

        if self.cuda:
            model = model.cuda()

        # set up optimizer
        opt_cfg = self.cfg['optimizer']
        optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg['learning_rate'],
                                    momentum=opt_cfg['momentum'], nesterov=opt_cfg['use_nesterov'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt_cfg['lr_annealing'])

        # primary training loop
        best_wer = math.inf

        for epoch in range(self.cfg['epochs']):
            # adjust lr
            scheduler.step()
            print("Learning rate annealed to {0:.6f}".format(scheduler.get_lr()[0]))

            avg_loss = self.train_epoch(train_loader, model, optimizer, epoch)
            print('Training Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\t'.format(epoch + 1, loss=avg_loss))

            avg_wer, avg_cer = validate(eval_loader, model)
            print('Validation Summary Epoch: [{0}]\tAverage WER {wer:.3f}\tAverage CER {cer:.3f}'
                  .format(epoch + 1, wer=avg_wer, cer=avg_cer))

            if avg_wer < best_wer:
                best_wer = avg_wer
                print("Better model found. Saving.")
                # model.serialize(self.output['model_path'])

    def train_epoch(self, train_loader, model, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        model.train()

        end = time.time()
        for i, data in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # create variables
            feat, target, feat_len, target_len = tuple(torch.autograd.Variable(i, requires_grad=False) for i in data)
            if self.cuda:
                feat = feat.cuda()

            # compute output
            output, output_len = model(feat, feat_len)
            loss = model.loss(output, target, output_len, target_len)

            # munge the loss
            avg_loss = loss.data.sum() / feat.size(0)  # average the loss by minibatch
            inf = math.inf
            if avg_loss == inf or avg_loss == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                avg_loss = 0
            losses.update(avg_loss, feat.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            if self.max_norm:
                torch.nn.utils.clip_grad_norm(model.parameters(), self.max_norm)
            optimizer.step()

            del loss
            del output
            del output_len

            # measure time taken
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format((epoch + 1), (i + 1), len(train_loader),
                                                                  batch_time=batch_time, data_time=data_time,
                                                                  loss=losses))
        return losses.avg

    @classmethod
    def load(cls, trainer_config, tqdm=False):
        try:
            cfg = TrainerConfiguration().load(trainer_config)
        except ValidationError as err:
            print(err.messages)
            raise err
        return cls(cfg.data, tqdm=tqdm)


def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    wers = AverageMeter()
    cers = AverageMeter()

    model.eval()

    end = time.time()
    for i, data in enumerate(val_loader):
        # create variables
        feat, target, feat_len, target_len = tuple(torch.autograd.Variable(i, volatile=True) for i in data)
        if model.is_cuda:
            feat = feat.cuda()

        # compute output
        output, output_len = model(feat, feat_len)
        loss = model.loss(output, target, output_len, target_len)

        # munge the loss
        avg_loss = loss.data.sum() / feat.size(0)  # average the loss by minibatch
        inf = math.inf
        if avg_loss == inf or avg_loss == -inf:
            print("WARNING: received an inf loss, setting loss value to 0")
            avg_loss = 0
        losses.update(avg_loss, feat.size(0))

        del loss
        del output
        del output_len

        # measure time taken
        batch_time.update(time.time() - end)
        end = time.time()

    return wers.avg, cers.avg, losses.avg
