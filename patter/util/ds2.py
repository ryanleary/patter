import re


def convert_state_dict(package):
    bidirectional = "lookahead.0.weight" not in package['state_dict']
    explicit_mapping = {}
    if bidirectional:
        explicit_mapping["fc.0.module.0.weight"] = "output.0.module.weight"
        explicit_mapping["fc.0.module.0.bias"] = "output.0.module.bias"
        explicit_mapping["fc.0.module.0.running_mean"] = "output.0.module.running_mean"
        explicit_mapping["fc.0.module.0.running_var"] = "output.0.module.running_var"
        explicit_mapping["fc.0.module.1.weight"] = "output.1.weight"
    else:
        explicit_mapping["fc.0.module.0.weight"] = "output.2.module.weight"
        explicit_mapping["fc.0.module.0.bias"] = "output.2.module.bias"
        explicit_mapping["fc.0.module.0.running_mean"] = "output.2.module.running_mean"
        explicit_mapping["fc.0.module.0.running_var"] = "output.2.module.running_var"
        explicit_mapping["lookahead.0.weight"] = "output.0.weight"
        explicit_mapping["fc.0.module.1.weight"] = "output.3.weight"
    cnn_types = {
        0: "cnn",
        1: "bn",
        2: "act"
    }
    blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                 'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
    state_dict = {}
    for k, v in package['state_dict'].items():
        if k in blacklist:
            continue
        if k.startswith("rnns."):
            k = "rnn." + k
        if k.startswith("conv"):
            parts = k.split(".")
            idx = int(parts[1])
            layer_idx = idx // 3
            layer_type = idx % 3
            parts[1] = str(layer_idx) + "-" + cnn_types[layer_type]
            k = ".".join(parts)
        k = re.sub(r'rnn\.rnns\.(\d+)\.batch_norm\.module\.(.*)', r'rnn.rnns.\1.batch_norm.\2', k)
        if k in explicit_mapping:
            k = explicit_mapping[k]
        state_dict[k] = v
    return state_dict


def generate_config(package):
    bidirectional = package['bidirectional'] if 'bidirectional' in package else True
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
            'bidirectional': bidirectional,
            'layers': package['hidden_layers'],
            'noise': None,
            'rnn_type': package['rnn_type'],
            'size': package['hidden_size']
        }
    }
    if not bidirectional:
        config['ctx'] = {
            "context": package['state_dict']['lookahead.0.weight'].shape[1] - 1,
            "activation": "hardtanh",
            "activation_params": [0, 20]
        }
    return config
