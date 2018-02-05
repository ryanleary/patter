from marshmallow import Schema, fields, post_load

class TrainerOutputConfiguration(Schema):
    model_path = fields.String(required=True)
    log_path = fields.String(required=True)


class TrainingSettings(Schema):
    epochs = fields.Integer(required=True)
    batch_size = fields.Integer(required=True)
    num_workers = fields.Integer(required=True)
    max_norm = fields.Float(required=False, default=400.0)


class OptimizerSettings(Schema):
    optimizer = fields.String(required=True)


class TrainerConfiguration(Schema):
    expt_id = fields.String(required=True, load_from="id")
    cuda = fields.Boolean(default=True)

    output = fields.Nested(TrainerOutputConfiguration)
    trainer = fields.Nested(TrainingSettings)
    optimizer = fields.Nested(OptimizerSettings, attribute="trainer.optimizer")
