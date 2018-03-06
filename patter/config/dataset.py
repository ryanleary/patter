from marshmallow import Schema, fields


class DatasetConfig(Schema):
    manifest = fields.String(required=True)
    name = fields.String(required=True)
    augment = fields.Boolean(required=False, default=False, missing=False)
