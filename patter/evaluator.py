import math
import time
import torch
from tqdm import tqdm as tqdm_wrap
from marshmallow.exceptions import ValidationError
from torch.utils.data import DataLoader

from patter.config import EvaluatorConfiguration
from patter.data import audio_seq_collate_fn
from patter.decoder import GreedyCTCDecoder
from patter.util import AverageMeter, split_targets


class Evaluator(object):
    def __init__(self, cfg, tqdm=False):
        self.cfg = cfg
        self.cuda = cfg['cuda']
        self.tqdm=tqdm

    def eval(self, model, corpus):
        test_loader = DataLoader(corpus, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                 pin_memory=True, batch_size=self.cfg['batch_size'])

        if self.cuda:
            model = model.cuda()

        wer, cer = validate(test_loader, model, decoder=None, tqdm=self.tqdm)
        return wer, cer

    @classmethod
    def load(cls, evaluator_config, tqdm=False):
        try:
            cfg = EvaluatorConfiguration().load(evaluator_config)
        except ValidationError as err:
            print(err.messages)
            raise err
        return cls(cfg.data, tqdm=tqdm)


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
