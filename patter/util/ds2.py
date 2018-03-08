import re


def convert_state_dict(package):
    explicit_mapping = {
        "fc.0.module.0.bias": "rnn.batch_norm.bias",
        "fc.0.module.0.running_mean": "rnn.batch_norm.running_mean",
        "fc.0.module.0.running_var": "rnn.batch_norm.running_var",
        "fc.0.module.0.weight": "rnn.batch_norm.weight",
        "fc.0.module.1.weight": "output.0.weight"
    }
    cnn_types = {
        0: "cnn",
        1: "batch_norm",
        2: "act"
    }
    state_dict = {}
    for k, v in package['state_dict'].items():
        if k.startswith("rnns."):
            k = "rnn." + k
        if k.startswith("conv"):
            parts = k.split(".")
            idx = int(parts[1])
            layer_idx = idx // 3
            layer_type = idx % 3
            parts[1] = str(layer_idx)
            parts.insert(2, cnn_types[layer_type])
            k = ".".join(parts)
        k = re.sub(r'rnn\.rnns\.(\d+)\.batch_norm\.module\.(.*)', r'rnn.rnns.\1.batch_norm.\2', k)
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
