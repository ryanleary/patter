from .greedy import GreedyCTCDecoder
from .beam import BeamCTCDecoder

valid_decoders = {
    "greedy": GreedyCTCDecoder,
    "beam": BeamCTCDecoder
}


class DecoderFactory(object):
    @classmethod
    def create(cls, cfg, labels, decoder_type=None):
        if decoder_type is None:
            decoder_type = cfg['algorithm']
        klass = valid_decoders[decoder_type]
        return klass.from_config(cfg, labels)
