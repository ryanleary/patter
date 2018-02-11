import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class SpeechCorpus(object):
    @classmethod
    def load(cls, config):
        pass


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


class SpectrogramDataset(Dataset):
    def __init__(self, manifest_filepath, labels, max_duration=None, min_duration=None):
        """
        Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations
        (in seconds). Each new line is a different sample. Example below:

        {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        ids = []
        duration = 0.0
        filtered_duration = 0.0
        with open(manifest_filepath) as fh:
            for line in fh:
                data = json.loads(line)
                if min_duration is not None and data['duration'] < min_duration:
                    filtered_duration += data['duration']
                    continue
                if max_duration is not None and data['duration'] > max_duration:
                    filtered_duration += data['duration']
                    continue
                ids.append(data)
                duration += data['duration']
        print("Dataset loaded with", duration/3600, "hours. Filtered", filtered_duration/3600, "hours.")
        self.ids = ids
        self.size = len(ids)
        self.duration = duration
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment_config)

    def __getitem__(self, index):
        sample = self.ids[index]
        spect = self.parse_audio(sample['audio_filepath'])
        transcript = self.parse_transcript(sample['text_filepath'])
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size
