import numpy as np
import torch
from .manifest import Manifest
from .features import PerturbedSpectrogramFeaturizer
from patter.config import CorporaConfiguration
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from marshmallow.exceptions import ValidationError


def audio_seq_collate_fn(batch):
    # sort batch by descending sequence length (for packed sequences later)
    batch.sort(key=lambda x: -x[0].size(1))
    minibatch_size = len(batch)

    # init tensors we need to return
    inputs = torch.zeros(minibatch_size, 1, batch[0][0].size(0), batch[0][0].size(1))
    input_lengths = torch.IntTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []

    # iterate over minibatch to fill in tensors appropriately
    for i, sample in enumerate(batch):
        input_lengths[i] = sample[0].size(1)
        inputs[i][0].narrow(1, 0, sample[0].size(1)).copy_(sample[0])
        target_sizes[i] = len(sample[1])
        targets.extend(sample[1])
    targets = torch.IntTensor(targets)
    return inputs, targets, input_lengths, target_sizes


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)


class AudioDataset(Dataset):
    def __init__(self, manifest_filepath, labels, featurizer, max_duration=None, min_duration=None):
        """
        Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations
        (in seconds). Each new line is a different sample. Example below:

        {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
        ...

        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param featurizer: Initialized featurizer class that converts paths of audio to feature tensors
        :param max_duration: If audio exceeds this length, do not include in dataset
        :param min_duration: If audio is less than this length, do not include in dataset
        """
        self.manifest = Manifest(manifest_filepath, max_duration=max_duration, min_duration=min_duration)
        print("Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours.".format(self.manifest.duration/3600,
                                                                                  self.manifest.filtered_duration/3600))
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.featurizer = featurizer

    def __getitem__(self, index):
        sample = self.manifest[index]
        features = self.featurizer.process(sample['audio_filepath'])
        transcript = self.parse_transcript(sample['text_filepath'])
        return features, transcript

    def __len__(self):
        return len(self.manifest)

    def get_largest_minibatch(self, minibatch_size):
        longest_sample = self.featurizer.max_augmentation_length(self[-1][0] + 20) # +20 gives some wiggle room
        freq_size = longest_sample.size(0)
        max_seqlength = longest_sample.size(1)
        targets = torch.IntTensor(max_seqlength*minibatch_size)
        feats = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        input_lengths = torch.IntTensor(minibatch_size)
        input_lengths.fill_(max_seqlength)
        return feats, targets, input_lengths, input_lengths

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    @classmethod
    def from_config(cls, corpus_config, feature_config, labels, manifest="train"):
        try:
            cfg = CorporaConfiguration().load(corpus_config)
        except ValidationError as err:
            raise err
        config = cfg.data
        datasets = {x['name']: x for x in config['datasets']}
        if manifest not in datasets:
            raise KeyError("Requested dataset ({}) doesn't exist.".format(manifest))
        dataset = datasets[manifest]

        augmentation_config = config['augmentation'] if dataset['augment'] else []
        featurizer = PerturbedSpectrogramFeaturizer.from_config(feature_config,
                                                                perturbation_configs=augmentation_config)

        return cls(dataset['manifest'], labels, featurizer, max_duration=config['max_duration'],
                   min_duration=config['min_duration'])
