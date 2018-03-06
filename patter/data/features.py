import scipy
import torch
import librosa
from .perturb import AudioAugmentor
from .segment import AudioSegment

windows = {
    'hann': scipy.signal.hann,
    'hamming': scipy.signal.hamming,
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
}


class PerturbedSpectrogramFeaturizer(object):
    def __init__(self, input_cfg, augmentor=None):
        self.augmentor = augmentor if augmentor is not None else AudioAugmentor()
        self.cfg = input_cfg
        self.window = windows.get(self.cfg['window'], windows['hamming'])

    def max_augmentation_length(self, length):
        return self.augmentor.max_augmentation_length(length)

    def process(self, file_path):
        audio = AudioSegment.from_file(file_path, target_sr=self.cfg['sample_rate'], int_values=self.cfg['int_values'])
        self.augmentor.perturb(audio)

        n_fft = int(self.cfg['sample_rate'] * self.cfg['window_size'])
        hop_length = int(self.cfg['sample_rate'] * self.cfg['window_stride'])
        dfft = librosa.stft(audio.samples, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=self.window)
        spect, _ = librosa.magphase(dfft)
        spect = torch.FloatTensor(spect).log1p()
        if self.cfg['normalize']:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect

    @classmethod
    def from_config(cls, input_config, perturbation_configs=None):
        if perturbation_configs is not None:
            aa = AudioAugmentor.from_config(perturbation_configs)
        else:
            aa = None

        return cls(input_config, augmentor=aa)