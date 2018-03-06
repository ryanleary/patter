from marshmallow import Schema, fields
from marshmallow.validate import Length


class LabelConfiguration(Schema):
    labels = fields.List(fields.String)


class NormalDistributionConfiguration(Schema):
    mean = fields.Float(default=0.0, missing=0.0)
    std = fields.Float(default=0.0, missing=0.001)


class RNNConfiguration(Schema):
    rnn_type = fields.String(default="lstm", load_from="type")
    bidirectional = fields.Boolean(default=True)
    size = fields.Integer(default=512)
    layers = fields.Integer(default=4)
    noise = fields.Nested(NormalDistributionConfiguration, default=None, missing=None)
    batch_norm = fields.Boolean(default=False, missing=False)


class CNNConfiguration(Schema):
    filters = fields.Integer(default=32)
    kernel = fields.List(fields.Integer, default=[21, 11], validate=Length(equal=2))
    stride = fields.List(fields.Integer, default=[2, 1], validate=Length(equal=2))
    padding = fields.List(fields.Integer, allow_none=True, validate=Length(equal=2))
    batch_norm = fields.Boolean(default=False)
    activation = fields.String(default="hardtanh")
    activation_params = fields.List(fields.Field, default=[], missing=[])


class ContextConfiguration(Schema):
    context = fields.Integer(default=20)
    activation = fields.String(default="hardtanh")
    activation_params = fields.List(fields.Field, default=[], missing=[])


class InputConfiguration(Schema):
    feat_type = fields.String(required=True, default="stft", missing="stft", load_from="type")
    normalize = fields.Boolean(default=True, missing=True)
    sample_rate = fields.Int(default=16000)
    window_size = fields.Float(default=0.02)
    window_stride = fields.Float(default=0.01)
    window = fields.String(default="hamming")
    int_values = fields.Boolean(default=False, missing=False)


class SpeechModelConfiguration(Schema):
    model = fields.String(required=True)

    input = fields.Nested(InputConfiguration, load_from="input")
    cnn = fields.Nested(CNNConfiguration, load_from="cnn", many=True)
    rnn = fields.Nested(RNNConfiguration, load_from="rnn")
    ctx = fields.Nested(ContextConfiguration, load_from="context")
    labels = fields.Nested(LabelConfiguration)
