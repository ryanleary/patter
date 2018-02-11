import os
from pathlib import Path
from tensorboardX import SummaryWriter
from datetime import datetime
import socket


def to_np(x):
    return x.data.cpu().numpy()


class TensorboardLogger(object):
    def __init__(self, _id, log_dir, model=None):
        expt_name = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + '_' + _id)
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
        self._id = _id
        self._model = model

    def log_step(self, step, loss):
        self._writer.add_scalar("loss_step/train", loss, step + 1)

    def init_epoch(self, loss, wer, cer, eval_loss):
        self._writer.add_scalar("loss/train", loss, 0)
        self._writer.add_scalar("loss/val", eval_loss, 0)
        self._writer.add_scalar("accuracy/wer", wer, 0)
        self._writer.add_scalar("accuracy/cer", cer, 0)

    def log_epoch(self, epoch, loss_results, wer_results, cer_results, eval_loss=None):
        self._writer.add_scalar("loss/train", loss_results[epoch], epoch + 1)
        self._writer.add_scalar("loss/val", eval_loss, epoch + 1)
        self._writer.add_scalar("accuracy/wer", wer_results[epoch], epoch + 1)
        self._writer.add_scalar("accuracy/cer", cer_results[epoch], epoch + 1)
        if self._model:
            for tag, value in self._model.named_parameters():
                tag = tag.replace('.', '/')
                self._writer.add_histogram(tag, to_np(value), epoch + 1)
                self._writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)

    def log_previous_epochs(self, end_epoch, loss_results, wer_results, cer_results):
        for i in range(end_epoch):
            values = {
                'Avg Train Loss': loss_results[i],
                'Avg WER': wer_results[i],
                'Avg CER': cer_results[i]
            }
            self._writer.add_scalars(self._id, values, i + 1)
