import numpy as np
import json
import soundfile
import random


class Perturbation(object):
    def perturb(self, data):
        raise NotImplementedError


class SpeedPerturbation(Perturbation):
    def __init__(self, min_rate=0.85, max_rate=1.15, rng=None):
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        speed_rate = self._rng.uniform(self._min_rate, self._max_rate)
        if speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")
        old_length = data._samples.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        data._samples = np.interp(new_indices, old_indices, data._samples)


class GainPerturbation(Perturbation):
    def __init__(self, min_gain_dbfs=-10, max_gain_dbfs=10, rng=None):
        self._min_gain_dbfs = min_gain_dbfs
        self._max_gain_dbfs = max_gain_dbfs
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        gain = self._rng.uniform(self._min_gain_dbfs, self._max_gain_dbfs)
        data._samples *= 10.**(gain / 20.)


class AudioAugmentor(object):
    def __init__(self, perturbations=None, rng=None):
        self._rng = random.Random() if rng is None else rng
        self._pipeline = perturbations if perturbations is not None else []

    def perturb(self, segment):
        for (prob, p) in self._pipeline:
            if self._rng.random() < prob:
                p.perturb(segment)
        return

    @classmethod
    def from_config(cls, config_file):
        config = []
        with open(config_file, "r") as fh:
            raw_config = json.load(fh)
        for p in raw_config:
            if p['type'] not in perturbations:
                print(p['type'], "perturbation not known. Skipping.")
                continue
            perturbation = perturbations[p['type']]
            config.append((p['prob'], perturbation(**p['params'])))
        return cls(perturbations=config)


perturbations = {"speed": SpeedPerturbation, "gain": GainPerturbation}


class AudioSegment(object):
    """Monaural audio segment abstraction.
    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate):
        """Create audio segment from samples.
        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        self._samples = self._convert_samples_to_float32(samples)
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        return ("%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, "
                "rms=%.2fdB" % (type(self), self.num_samples, self.sample_rate,
                                self.duration, self.rms_db))

    def _convert_samples_to_float32(self, samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2**(bits - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    @classmethod
    def from_file(cls, filename):
        samples, sample_rate = soundfile.read(filename, dtype='float32')
        return cls(samples, sample_rate)

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_samples(self):
        return self._samples.shape[0]

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        mean_square = np.mean(self._samples**2)
        return 10 * np.log10(mean_square)
