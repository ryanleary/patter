"""
[corpora]
train_manifest = "/path/to/corpora/train.json"
val_manifest = "/path/to/corpora/val.json"


[corpora.config]
min_duration = 1.0
max_duration = 17.0


[[corpora.augmentation]]
type = "noise"
prob = 0.6
[corpora.augmentation.cfg]
manifest = "/path/to/noise/manifest.json"
min_snr_db = 40
max_snr_db = 50

[[corpora.augmentation]]
type = "impulse"
prob = 0.5
[corpora.augmentation.cfg]
manifest = "/path/to/impulse/manifest.json"

[[corpora.augmentation]]
type = "speed"
prob = 0.5
[corpora.augmentation.cfg]
min_speed_rate = 0.95
max_speed_rate = 1.05

[[corpora.augmentation]]
type = "shift"
prob = 1.0
[corpora.augmentation.cfg]
min_shift_ms = -5
max_shift_ms = 5

[[corpora.augmentation]]
type = "volume"
prob = 0.2
[corpora.augmentation.cfg]
min_gain_dbfs = -10
max_gain_dbfs = 10
"""

from marshmallow import Schema, fields


class AugmentationSpec(Schema):
    aug_type = fields.String(load_from="type", required=True)
    prob = fields.Float(required=True, )


class CorporaConfig(Schema):
    min_duration = fields.Float(missing=None)
    max_duration = fields.Float(missing=None)


class CorporaConfiguration(Schema):
    train_manifest = fields.String(required=True)
    val_manifest = fields.String(required=True)

    config = fields.Nested(CorporaConfig)
    augmentation = fields.List(AugmentationConfig)
