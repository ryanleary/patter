from marshmallow import Schema, fields

from .dataset import DatasetConfig


class DecoderConfig(Schema):
    algorithm = fields.String(required=True)
    num_workers = fields.Integer(default=4, missing=4)


class EvaluatorConfiguration(Schema):
    datasets = fields.Nested(DatasetConfig, load_from="dataset", many=True)
    decoder = fields.Nested(DecoderConfig)

    cuda = fields.Boolean(default=True)
    batch_size = fields.Integer(required=True)
    num_workers = fields.Integer(required=True)
