import unittest
from patter.config import corpora
from patter.data import AudioSegment
from patter.data.perturb import AudioAugmentor
import toml


class TestAudioAugment(unittest.TestCase):
    def test_aa(self):
        config_path = "../../../etc/corpus_config.toml"
        with open(config_path, "r") as fh:
            corpus_config = toml.load(fh)
        try:
            cfg = corpora.CorporaConfiguration().load(corpus_config)
        except Exception as err:
            print(err.messages)
            raise err

        aug_config = cfg.data['augmentation']
        aa = AudioAugmentor.from_config(aug_config)
        test_audio = AudioSegment.from_file("../../../../patter_data/speech/908-31957-0019.wav")
        aa.perturb(test_audio)
        self.assertTrue(False)