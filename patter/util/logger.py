import os
from pathlib import Path
from tensorboardX import SummaryWriter
from datetime import datetime
import socket
from collections import defaultdict


def to_np(x):
    return x.data.cpu().numpy()


class TensorboardLogger(object):
    def __init__(self, tb_id, log_dir, include_grad=False):
        expt_name = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + '_' + tb_id
        log_dir = Path(log_dir) / expt_name

        try:
            log_dir.mkdir(parents=True)
        except FileExistsError as e:
            print('Tensorboard log directory already exists.')
            for file in os.listdir(str(log_dir)):
                file_path = os.path.join(str(log_dir), file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception:
                    raise

        self._writer = SummaryWriter(str(log_dir))
        self._id = tb_id
        self._include_grad = include_grad

    def log_step(self, step, loss):
        self._writer.add_scalar("loss_step/train", loss, step)

    def log_images(self, epoch, images):
        for tup in images:
            self._writer.add_image(tup[0], tup[1], epoch)

    def add_graph(self, model, dummy_input):
        self._writer.add_graph(model, dummy_input)

    def log_sample_decodes(self, epoch, sample_decodes):
        table = defaultdict(list)
        table[0].append("*")
        table[1].append("-")
        table[2].append("Hyp")
        table[3].append("Ref")
        for x in range(len(sample_decodes)):
            table[0].append("Ex. {}".format(x))
            table[1].append("-")
            table[2].append(sample_decodes[x][0])
            table[3].append(sample_decodes[x][1])

        final_table = "\n".join(["|".join(table[0]), "|".join(table[1]), "|".join(table[2]), "|".join(table[3])])
        self._writer.add_text("Validation Decode", final_table, epoch)

    def log_epoch(self, epoch, loss, wer, cer, model=None):
        self._writer.add_scalar("loss/train", loss, epoch)
        self._writer.add_scalar("accuracy/wer", wer, epoch)
        self._writer.add_scalar("accuracy/cer", cer, epoch)

        if model:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                self._writer.add_histogram(tag, to_np(value), epoch + 1)
                if self._include_grad:
                    self._writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)

    def log_previous_epochs(self, end_epoch, loss_results, wer_results, cer_results):
        for i in range(end_epoch):
            values = {
                'Avg Train Loss': loss_results[i],
                'Avg WER': wer_results[i],
                'Avg CER': cer_results[i]
            }
            self._writer.add_scalars(self._id, values, i + 1)
