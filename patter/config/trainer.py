from marshmallow import Schema, fields


class TrainerOutputConfiguration(Schema):
    model_path = fields.String(required=True)
    log_path = fields.String(required=True)


class OptimizerSettings(Schema):
    optimizer = fields.String(required=True)
    learning_rate = fields.Float(required=True, load_from="lr")
    momentum = fields.Float(default=0.9)
    use_nesterov = fields.Boolean(default=True, missing=True)
    lr_annealing = fields.Float(default=1.1, load_from="anneal")


class TrainingSettings(Schema):
    epochs = fields.Integer(required=True)
    batch_size = fields.Integer(required=True)
    num_workers = fields.Integer(required=True)
    max_norm = fields.Float(required=False, default=400.0)
    optimizer = fields.Nested(OptimizerSettings)


class TrainerConfiguration(Schema):
    expt_id = fields.String(required=True, load_from="id")
    cuda = fields.Boolean(default=True)

    output = fields.Nested(TrainerOutputConfiguration)
    trainer = fields.Nested(TrainingSettings)
