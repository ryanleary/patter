from marshmallow import Schema, fields


class TrainerOutputConfiguration(Schema):
    model_path = fields.String(required=True)
    log_path = fields.String(required=True)
    checkpoint_interval = fields.Integer(default=0, missing=0)


class OptimizerSettings(Schema):
    optimizer = fields.String(required=True)
    lr = fields.Float(required=True)
    momentum = fields.Float(default=0.9)
    nesterov = fields.Boolean(default=True)


class SchedulerSettings(Schema):
    lr_annealing = fields.Float(default=1.1, load_from="anneal")


class TrainingSettings(Schema):
    epochs = fields.Integer(required=True)
    batch_size = fields.Integer(required=True)
    num_workers = fields.Integer(required=True)
    max_norm = fields.Float(required=False, default=400.0)
    optimizer = fields.Nested(OptimizerSettings)
    scheduler = fields.Nested(SchedulerSettings)


class TrainerConfiguration(Schema):
    expt_id = fields.String(required=True, load_from="id")
    cuda = fields.Boolean(default=True)

    output = fields.Nested(TrainerOutputConfiguration)
    trainer = fields.Nested(TrainingSettings)
