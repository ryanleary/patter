# patter
speech-to-text framework in PyTorch with initial support for the DeepSpeech2 architecture (and variants of it). 

## Features
- File-based configuration of corpora definitions, model architecture, and training configuration for repeatability
- DeepSpeech model is highly configurable
  - Various RNN types (RNN, LSTM, GRU) and sizes (layers/hidden units)
  - Various activation functions (Clipped ReLU, Swish)
  - Forward-only RNN with Lookahead (for streaming) or Bidirectional RNN
  - Configurable CNN frontend
  - Optional batchnorm
  - Optional RNN weight noise
- Beam decoder with KenLM support
- Dataset augmentation with support for:
  - speed perturbations
  - gain perturbations
  - shift (in time) perturbations
  - noise addition (at random SNR)
  - impulse response perturbations
- Tensorboard integration

## Installation
Manual installation of two dependencies is required:
- [SeanNaren/warp-ctc](https://github.com/seannaren/warp-ctc) and the pytorch binding included within the repo
- [parlance/ctcdecode](https://github.com/parlance/ctcdecode) CTC beam decoder enabling language model support

Once these dependencies are installed, patter can be installed by simply running `python setup.py install`. For
debugging and development purposes, patter can instead be installed with `python setup.py develop`.

## Dataset Definitions
Datasets for patter are defined using json-lines files with newline-separated json objects. Each link contains a json
object which defines an utterance's audio path, transcription path, and duration in seconds.

```json
{"audio_filepath": "/path/to/utterance1.wav", "text_filepath": "/path/to/utterance1.txt", "duration": 23.147}
{"audio_filepath": "/path/to/utterance2.wav", "text_filepath": "/path/to/utterance2.txt", "duration": 18.251}
```

## Training
Patter includes a top-level trainer script which calls to underlying library methods for training. To use the built-in
command-line trainer, three files must be defined: corpus configuration, model configuration, and training configuration.
Examples for each are provided below.

### Corpus Configuration
A corpus configuration file is used to specify the train and validation sets in the corpus as well as any augmentation
that should occur to the audio. See the example configuration below for further documentation on the options.

```toml
# Filter the audio configured in the `datasets` below to be within min and max duration. Remove min or max (or both) to
# do no filtering
min_duration = 1.0
max_duration = 17.0

# Link to manifest files (as described above) of the training and validation sets. A future release will allow multiple
# files to be specified for merging corpora on the fly. If `augment` is true, each audio will be passed through the 
# augmentation pipeline specified below. Valid names for the datasets are in the set ["train", "val"]
[[dataset]]
name = "train"
manifest = "/path/to/corpora/train.json"
augment = true

[[dataset]]
name = "val"
manifest = "/path/to/corpora/val.json"
augment = false


# Optional augmentation pipeline. If specified, audio from a dataset with the augment flag set to true will be passed
# through each augmentation, in order. Each augmentation must minimally specify the type and a probability. The 
# probability indicates that the augmentation will run on a given audio file with that probability

# The noise augmentation mixes audio from a dataset of noise files with a random SNR drawn from within the range specified.
[[augmentation]]
type = "noise"
prob = 0.0
[augmentation.config]
manifest = "/path/to/noise_manifest.json"
min_snr_db = 3
max_snr_db = 35

# The impulse augmentation applies a random impulse response drawn from the manifest to the audio 
[[augmentation]]
type = "impulse"
prob = 0.0
[augmentation.config]
manifest = "/path/to/impulse_manifest.json"

# The speed augmentation applies a random speed perturbation without altering pitch
[[augmentation]]
type = "speed"
prob = 1.0
[augmentation.config]
min_speed_rate = 0.95
max_speed_rate = 1.05

# The shift augmentation simply adds a random amount of silence to the audio or removes some of the initial audio
[[augmentation]]
type = "shift"
prob = 1.0
[augmentation.config]
min_shift_ms = -5
max_shift_ms = 5

# The gain augmentation modifies the gain of the audio by a fixed amount randomly chosen within the specified range
[[augmentation]]
type = "gain"
prob = 1.0
[augmentation.config]
min_gain_dbfs = -10
max_gain_dbfs = 10
```

### Model Configuration
At this time, patter supports only variants of the DeepSpeech 2 and DeepSpeech 3 (same as DS2 w/o BatchNorm + Weight Noise)
architectures. Future model architectures including novel architectures may be included in future releases. To configure
the architecture and hyperparameters, define the model a configuration TOML. See example:

```toml
# model class - only DeepSpeechOptim currently
model = "DeepSpeechOptim"

# define input features/windowing. Currently only STFT is supported, but window is configurable.
[input]
type = "stft"
normalize = true
sample_rate = 16000
window_size = 0.02
window_stride = 0.01
window = "hamming"

# Define layers of [2d CNN -> Activation -> Optional BatchNorm] as a frontend
[[cnn]]
filters = 32
kernel = [41, 11]
stride = [2, 2]
padding = [0, 10]
batch_norm = true
activation = "hardtanh"
activation_params = [0, 20]

[[cnn]]
filters = 32
kernel = [21, 11]
stride = [2, 1]
padding = [0, 2]
batch_norm = true
activation = "hardtanh"
activation_params = [0, 20]

# Configure the RNN. Currently LSTM, GRU, and RNN are supported. QRNN will be added for forward-only models in a future release
[rnn]
type = "lstm"
bidirectional = true
size = 512
layers = 4
batch_norm = true

# DS3 suggests using weight noise instead of batch norm, only set when rnn batch_norm = false
#[rnn.noise]
#mean=0.0
#std=0.001

# only used/necessary when rnn bidirectional = false
#[context]
#context = 20
#activation = "swish"

# Set of labels for model to predict. Specifying a label for the CTC 'blank' symbol is not required and handled automatically
[labels]
labels = [
  "'", "A", "B", "C", "D", "E", "F", "G", "H",
  "I", "J", "K", "L", "M", "N", "O", "P", "Q",
  "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " ",
]
```

### Trainer Configuration
The trainer configuration file includes metadata about the model to be created, where to store models, logs, tensorboard
logs, etc, in addition to NN trainer configuration.

```toml
# give the trained model a name
id = "expt-name"
cuda = true

[output]
model_path = "/path/to/best/model.pt"
log_path = "/path/to/tensorboard/logs"

[trainer]
epochs = 20
batch_size = 32
num_workers = 4
max_norm = 400

[trainer.optimizer]
# Currently SGD and Adam are supported
optimizer = "sgd"
lr = 3e-4
momentum = 0.9
anneal = 0.85
```

## Testing
A patter-test script is provided for doing evaluations of a trained model. It takes as arguments a testing configuration
and a trained model.

```toml
cuda = true
batch_size = 10
num_workers = 4

[[dataset]]
name = "test"
manifest = "/path/to/manifests/test.jl"
augment = false

[decoder]
algorithm = "greedy" # or "beam"
workers = 4

# If `beam` is specified as the decoder type, the below is used to initialize the beam decoder
[decoder.beam]
beam_width = 30
cutoff_top_n = 40
cutoff_prob = 1.0

# If "beam" is specified and you want to use a language model, configure the ARPA or KenLM format LM and alpha/beta weights
[decoder.beam.lm]
lm_path = "/path/to/language/model.arpa"
alpha = 2.15
beta = 0.35
```

## Acknowledgements
Huge thanks to [SeanNaren](https://github.com/seannaren) whose work on [deepspeech.pytorch](https://github.com/seannaren/deepspeech.pytorch) is leveraged heavily in this project.
