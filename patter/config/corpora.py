from marshmallow import Schema, fields
from marshmallow.validate import Range

from .dataset import DatasetConfig


class AugmentationConfig(Schema):
    manifest_path = fields.String(load_from="manifest")
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

    cfg = fields.Nested(AugmentationConfig, load_from="config")


class CorporaConfiguration(Schema):
    min_duration = fields.Float(missing=None)
    max_duration = fields.Float(missing=None)
    datasets = fields.Nested(DatasetConfig, load_from="dataset", many=True)
    augmentation = fields.Nested(AugmentationSpec, many=True, missing=[])
