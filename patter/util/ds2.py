import re


def convert_state_dict(package):
    explicit_mapping = {
        "fc.0.module.0.bias": "rnns.batch_norm.bias",
        "fc.0.module.0.running_mean": "rnns.batch_norm.running_mean",
        "fc.0.module.0.running_var": "rnns.batch_norm.running_var",
        "fc.0.module.0.weight": "rnns.batch_norm.weight",
        "fc.0.module.1.weight": "output.0.weight"
    }
    state_dict = {}
    for k, v in package['state_dict'].items():
        if k.startswith("rnns."):
            k = "rnns." + k
        k = re.sub(r'rnns\.rnns\.(\d+)\.batch_norm\.module\.(.*)', r'rnns.rnns.\1.batch_norm.\2', k)
        if k in explicit_mapping:
            k = explicit_mapping[k]
        state_dict[k] = v
    return state_dict


def generate_config(package):
    config = {
        'cnn': [
            {
                'activation': 'hardtanh',
                'activation_params': [0, 20],
                'batch_norm': True,
                'filters': 32,
                'kernel': [41, 11],
                'padding': [0, 10],
                'stride': [2, 2]
            }, {
                'activation': 'hardtanh',
                'activation_params': [0, 20],
                'batch_norm': True,
                'filters': 32,
                'kernel': [21, 11],
                'padding': [0, 0],
                'stride': [2, 1]
            }
        ],
        'input': {
            'feat_type': 'stft',
            'normalize': True,
            'int_values': True,
            'sample_rate': package['audio_conf']['sample_rate'],
            'window': package['audio_conf']['window'],
            'window_size': package['audio_conf']['window_size'],
            'window_stride': package['audio_conf']['window_stride']
        },
        'labels': {
            'labels': list(package['labels'][1:])
        },
        'model': 'DeepSpeechOptim',
        'rnn': {
            'batch_norm': True,
            'bidirectional': True,
            'layers': package['hidden_layers'],
            'noise': None,
            'rnn_type': package['rnn_type'],
            'size': package['hidden_size']
        }
    }
    return config
