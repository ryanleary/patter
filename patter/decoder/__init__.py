from .greedy import GreedyCTCDecoder
from .beam import BeamCTCDecoder

valid_decoders = {
    "greedy": GreedyCTCDecoder,
    "beam": BeamCTCDecoder
}