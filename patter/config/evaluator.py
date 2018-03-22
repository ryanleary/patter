from marshmallow import Schema, fields
from marshmallow.validate import OneOf

from .dataset import DatasetConfig
from patter.decoder import valid_decoders


class BeamLMDecoderConfig(Schema):
    lm_path = fields.String(required=True)
    alpha = fields.Float(required=True)
    beta = fields.Float(required=True)


class BeamDecoderConfig(Schema):
    beam_width = fields.Integer(required=True)
    cutoff_top_n = fields.Integer(default=40, missing=40)
    cutoff_prob = fields.Float(default=1.0, missing=1.0)

    lm = fields.Nested(BeamLMDecoderConfig)


class DecoderConfig(Schema):
    algorithm = fields.String(required=True, validate=OneOf(valid_decoders))
    num_workers = fields.Integer(default=4, missing=4)
    beam_config = fields.Nested(BeamDecoderConfig, load_from="beam")


class EvaluatorConfiguration(Schema):
    datasets = fields.Nested(DatasetConfig, load_from="dataset", many=True)
    decoder = fields.Nested(DecoderConfig)

    cuda = fields.Boolean(default=True)
    batch_size = fields.Integer(required=True)
    num_workers = fields.Integer(required=True)
