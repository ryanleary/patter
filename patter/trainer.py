import math
import time
import torch
from tqdm import tqdm as tqdm_wrap
from torch.utils.data import DataLoader
from marshmallow.exceptions import ValidationError
from .config import TrainerConfiguration
from .decoder import GreedyCTCDecoder
from .data import BucketingSampler, audio_seq_collate_fn
from .util import AverageMeter, TensorboardLogger
from .models import SpeechModel

optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam
}


class NoOpScheduler(object):
    def __init__(self):
        pass

    def step(self):
        pass


class Trainer(object):
    def __init__(self, train_config, tqdm=False):
        self.cfg = train_config['trainer']
        self.output = train_config['output']
        self.cuda = train_config['cuda']
        self.train_id = train_config['expt_id']
        self.tqdm=tqdm
        self.max_norm = self.cfg.get('max_norm', None)
        self.logger = TensorboardLogger(train_config['expt_id'], self.output['log_path'])

    def warmup(self, model, corpus, optimizer, batch_size):
        # warm up with the largest sized minibatch
        data = corpus.get_largest_minibatch(batch_size)
        feat, target, feat_len, target_len = tuple(torch.autograd.Variable(i, requires_grad=False) for i in data)
        if self.cuda:
            feat = feat.cuda(async=True)
        # self.logger.add_graph(model, (feat, feat_len))
        output, output_len = model(feat, feat_len)
        loss = model.loss(output, target, output_len, target_len)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        del feat
        del loss
        del output
        del output_len

    def train(self, model, corpus, eval=None):
        # set up data loaders
        train_sampler = BucketingSampler(corpus, batch_size=self.cfg['batch_size'])
        train_loader = DataLoader(corpus, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                  pin_memory=True, batch_sampler=train_sampler)
        if eval is not None:
            eval_loader = DataLoader(eval, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                     pin_memory=True, batch_size=self.cfg['batch_size'])
        else:
            eval_loader = None

        if self.cuda:
            model = model.cuda()

        print(model)

        # set up optimizer
        opt_cfg = self.cfg['optimizer']
        optim_class = optimizers.get(opt_cfg['optimizer'])
        del opt_cfg['optimizer']
        optimizer = optim_class(model.parameters(), **opt_cfg)
        print("Configured with optimizer:", optimizer)

        # set up a learning rate scheduler if requested -- currently only StepLR supported
        if "scheduler" in self.cfg:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.cfg['scheduler']['lr_annealing'])
            print("Configured with learning rate scheduler:", scheduler)
        else:
            scheduler = NoOpScheduler()

        # warm up gpu memory cache by doing a fwd/bwd, validation, and resetting model
        print("Starting warmup...")
        self.warmup(model, corpus, optimizer, self.cfg['batch_size'])
        avg_wer, avg_cer = validate(eval_loader, model)
        self.logger.log_epoch(0, 500, avg_wer, avg_cer, 500)
        print("Warmup complete.")

        # primary training loop
        best_wer = math.inf

        for epoch in range(self.cfg['epochs']):
            # shuffle the input data if required
            if epoch > 0:
                train_sampler.shuffle()

            # adjust lr
            scheduler.step()
            # print("> Learning rate annealed to {0:.6f}".format(scheduler.get_lr()[0]))
            
            avg_loss = self.train_epoch(train_loader, model, optimizer, epoch)
            print("Epoch {} Summary:".format(epoch))
            print('    Train:\tAverage Loss {loss:.3f}\t'.format(loss=avg_loss))

            avg_wer, avg_cer, val_loss, sample_decodes = validate(eval_loader, model, training=True, log_n_examples=10)
            print('    Validation:\tAverage WER {wer:.3f}\tAverage CER {cer:.3f}'
                  .format(wer=avg_wer, cer=avg_cer))

            # log the result of the epoch
            self.logger.log_epoch(epoch+1, avg_loss, avg_wer, avg_cer, val_loss, model=model)
            self.logger.log_images(epoch+1, model.get_filter_images())
            self.logger.log_sample_decodes(epoch+1, sample_decodes)

            if avg_wer < best_wer:
                best_wer = avg_wer
                # print("Better model found. Saving.")
                torch.save(SpeechModel.serialize(model, optimizer=optimizer), self.output['model_path'])

    def train_epoch(self, train_loader, model, optimizer, epoch):
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        loader = train_loader
        if self.tqdm:
            loader = tqdm_wrap(loader, desc="Epoch {}".format(epoch+1), leave=False)

        end = time.time()
        for i, data in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # create variables
            feat, target, feat_len, target_len = tuple(torch.autograd.Variable(i, requires_grad=False) for i in data)
            if self.cuda:
                feat = feat.cuda(async=True)

            # compute output
            # feat is (batch, 1,  feat_dim,  seq_len)
            # output is (seq_len, batch, output_dim)
            output, output_len = model(feat, feat_len)
            loss = model.loss(output, target, output_len, target_len)

            # munge the loss
            avg_loss = loss.data.sum() / feat.size(0)  # average the loss by minibatch
            inf = math.inf
            if avg_loss == inf or avg_loss == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                avg_loss = 0
            self.logger.log_step(epoch*len(train_loader) + i, avg_loss)
            losses.update(avg_loss, feat.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            if self.max_norm:
                torch.nn.utils.clip_grad_norm(model.parameters(), self.max_norm)
            optimizer.step()

            del feat
            del avg_loss
            del loss
            del output
            del output_len

            # measure time taken
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.tqdm:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format((epoch + 1), (i + 1), len(train_loader),
                                                                      batch_time=batch_time, data_time=data_time,
                                                                      loss=losses))
            else:
                loader.set_postfix(loss=losses.val)
        return losses.avg

    @classmethod
    def load(cls, trainer_config, tqdm=False):
        try:
            cfg = TrainerConfiguration().load(trainer_config)
        except ValidationError as err:
            print(err.messages)
            raise err
        return cls(cfg.data, tqdm=tqdm)


def split_targets(targets, target_sizes):
    results = []
    offset = 0
    for size in target_sizes:
        results.append(targets[offset:offset + size])
        offset += size
    return results


def validate(val_loader, model, decoder=None, tqdm=True, training=False, log_n_examples=0):
    if decoder is None:
        decoder = GreedyCTCDecoder(model.labels)
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    loader = tqdm_wrap(val_loader, desc="Validate", leave=False) if tqdm else val_loader

    end = time.time()
    wer, cer = 0.0, 0.0
    decodes = []
    refs = []
    for i, data in enumerate(loader):
        # create variables
        feat, target, feat_len, target_len = tuple(torch.autograd.Variable(i, volatile=True) for i in data)
        if model.is_cuda:
            feat = feat.cuda()

        # compute output
        output, output_len = model(feat, feat_len)

        if training:
            mb_loss = model.loss(output, target, output_len, target_len)
            avg_loss = mb_loss.data.sum() / feat.size(0)  # average the loss by minibatch
            inf = math.inf
            if avg_loss == inf or avg_loss == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                avg_loss = 0
            losses.update(avg_loss, feat.size(0))

        # do the decode
        decoded_output, _ = decoder.decode(output.transpose(0, 1).data, output_len.data)
        target_strings = decoder.convert_to_strings(split_targets(target.data, target_len.data))

        if len(decodes) < log_n_examples:
            decodes.append(decoded_output[0][0])
            refs.append(target_strings[0][0])

        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer += decoder.wer(transcript, reference) / float(len(reference.split()))
            cer += decoder.cer(transcript, reference) / float(len(reference))

        del output
        del output_len

        # measure time taken
        batch_time.update(time.time() - end)
        end = time.time()
    wer = wer * 100 / len(val_loader.dataset)
    cer = cer * 100 / len(val_loader.dataset)

    if training:
        return wer, cer, losses.avg, list(zip(decodes, refs))
    return wer, cer
