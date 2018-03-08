from marshmallow import Schema, fields
from marshmallow.validate import OneOf

from .dataset import DatasetConfig
from patter.decoder import valid_decoders


class DecoderConfig(Schema):
    algorithm = fields.String(required=True, validate=OneOf(valid_decoders))
    num_workers = fields.Integer(default=4, missing=4)


class EvaluatorConfiguration(Schema):
    datasets = fields.Nested(DatasetConfig, load_from="dataset", many=True)
    decoder = fields.Nested(DecoderConfig)

    cuda = fields.Boolean(default=True)
    batch_size = fields.Integer(required=True)
    num_workers = fields.Integer(required=True)
