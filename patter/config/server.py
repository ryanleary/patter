from marshmallow import Schema, fields

from .decoder import DecoderConfig


class ServerConfiguration(Schema):
    decoder = fields.Nested(DecoderConfig)

    cuda = fields.Boolean(default=True)
    batch_size = fields.Integer(required=True)

    model_path = fields.String(required=True)
    language = fields.String(required=True)