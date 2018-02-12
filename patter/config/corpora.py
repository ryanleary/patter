from marshmallow import Schema, fields
from marshmallow.validate import Range


class AugmentationConfig(Schema):
    manifest = fields.String()
    min_snr_db = fields.Float()
    max_snr_db = fields.Float()
    min_speed_rate = fields.Float()
    max_speed_rate = fields.Float()
    min_shift_ms = fields.Float()
    max_shift_ms = fields.Float()
    min_gain_dbfs = fields.Float()
    max_gain_dbfs = fields.Float()


class AugmentationSpec(Schema):
    aug_type = fields.String(load_from="type", required=True)
    prob = fields.Float(required=True, validate=Range(0, 1))

    cfg = fields.Nested(AugmentationConfig)


class CorporaConfig(Schema):
    min_duration = fields.Float(missing=None)
    max_duration = fields.Float(missing=None)


class CorporaConfiguration(Schema):
    train_manifest = fields.String(required=True)
    val_manifest = fields.String(required=True)

    config = fields.Nested(CorporaConfig)
    augmentation = fields.List(AugmentationSpec)
