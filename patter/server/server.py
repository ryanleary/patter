import grpc
import torch
import numpy as np
import collections
import time
import threading
from marshmallow.exceptions import ValidationError

from . import speech_pb2, speech_pb2_grpc

from patter import ModelFactory
from patter.config import ServerConfiguration
from patter.data import AudioSegment
from patter.decoder import DecoderFactory
from patter.data.features import PerturbedSpectrogramFeaturizer


class SpeechServicer(speech_pb2_grpc.SpeechServicer):
    def __init__(self, model_path, decoder_config, language="en-US", cuda=False):
        # initialize the model to test
        self._model = ModelFactory.load(model_path)
        self._decoder = DecoderFactory.create(decoder_config, self._model.labels)
        self._featurizer = PerturbedSpectrogramFeaturizer.from_config(self._model.input_cfg)
        self._language = language
        self._use_cuda = cuda

        if self._use_cuda:
            self._model = self._model.cuda()
        self._model.eval()

    def _raw_data_to_samples(self, data, sample_rate=16000, encoding=None):
        # TODO: support other encodings
        if sample_rate == 16000 and encoding == speech_pb2.RecognitionConfig.LINEAR16:
            signal = np.frombuffer(data, dtype=np.int16)
        else:
            raise ValueError("Unsupported audio data configuration")
            signal = None
        return signal

    def Recognize(self, request, context):
        print("Handling batch request.")
        config = request.config

        # check audio format (sample rate, encoding) to convert if necessary
        if config.language_code != self._language:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details('Requested unsupported language')
            return

        max_alternatives = max(min(config.max_alternatives, 30), 1)

        # get samples
        samples = self._raw_data_to_samples(request.audio.content, sample_rate=config.sample_rate_hertz, encoding=config.encoding)
        segment = AudioSegment(samples, config.sample_rate_hertz, target_sr=self._model.input_cfg['sample_rate'])

        # featurize
        features = self._featurizer.process_segment(segment)
        features = features.unsqueeze(0).unsqueeze(0)

        # run model
        output, output_len = self._model(torch.autograd.Variable(features, requires_grad=False),
                                         torch.autograd.Variable(torch.IntTensor([features.size(3)]), requires_grad=False))
        output = output.transpose(0, 1)

        # decode
        decoded_output, offsets, scores = self._decoder.decode(output.data, output_len.data, num_results=max_alternatives)

        # build output message
        alternatives = []
        transcripts = set([])
        for idx in range(min(max_alternatives, len(decoded_output[0]))):
            transcript = decoded_output[0][idx].strip().lower()
            if transcript not in transcripts:
                transcripts.add(transcript)
            else:
                continue
            transcript_words = transcript.split()
            words = []
            if idx == 0:
                for w in transcript_words:
                    words.append(speech_pb2.WordInfo(word=w))
            alternatives.append(speech_pb2.SpeechRecognitionAlternative(transcript=transcript, confidence=scores[0][idx], words=words))
        # may be multiple results if there are multiple chunks created
        results = [speech_pb2.SpeechRecognitionResult(alternatives=alternatives)]
        response = speech_pb2.RecognizeResponse(results=results)
        return response

    def StreamingRecognize(self, request_iterator, context):
        print("Handling stream request...")
        config_wrapper = request_iterator.next()
        if not config_wrapper.HasField("streaming_config"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details('First StreamingRequest must be a configuration request')
            return
            # return an error
        stream_config = config_wrapper.streaming_config

        # check audio format (sample rate, encoding) to convert if necessary
        if stream_config.config.language_code != self._language:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details('Requested unsupported language')
            return

        sample_buffer = collections.deque()
        done = False
        last_incoming = time.time()

        def read_incoming():
            try:
                while 1:
                    received = next(request_iterator)
                    samples = self._raw_data_to_samples(received.audio_content, sample_rate=stream_config.config.sample_rate_hertz, encoding=stream_config.config.encoding)
                    sample_buffer.extend(samples)
                    last_incoming = time.time()
            except StopIteration:
                print("reached end")
                return
            except ValueError:
                context.set_code(grpc.StatusCode.UNIMPLEMENTED)
                context.set_details('Unable to handle requested audio type')
                raise ValueError('Unable to handle requested audio type')

        thread = threading.Thread(target=read_incoming)
        thread.daemon = True
        thread.start()

        last_check = time.time()
        full_transcript = ""
        hidden = None
        result = None
        last_buffer_size = -1
        while 1:
            stream_done = time.time()-last_incoming > self._flush_time
            if len(sample_buffer) > self._min_buffer or (time.time()-last_check >= self._flush_time and len(sample_buffer) > self._min_buffer):
                last_check = time.time()
                signal = self._get_np_from_deque(sample_buffer, size=min(len(sample_buffer), self._max_buffer), reserve=int(0.4*self._model_sample_rate))
                spect = self._parser.parse_audio_data(signal).contiguous()
                spect = spect.view(1, 1, spect.size(0), spect.size(1))
                out, _ = self._model(torch.autograd.Variable(spect, volatile=True), hidden)
                out = out.transpose(0, 1)  # TxNxH
                decoded_output, _, _, _ = self._decoder.decode(out.data[:-19,:,:])
                full_transcript += decoded_output[0][0]
                alt = speech_pb2.SpeechRecognitionAlternative(transcript=full_transcript)
                result = speech_pb2.StreamingRecognitionResult(alternatives=[alt], is_final=done)
                out = speech_pb2.StreamingRecognizeResponse(results=[result])
                # if stream_done:
                #     return out
                yield out
            else:
                last_check = time.time()
                time.sleep(0.01)



    @classmethod
    def from_config(cls, server_config):
        try:
            cfg = ServerConfiguration().load(server_config)
            if len(cfg.errors) > 0:
                raise ValidationError(cfg.errors)
        except ValidationError as err:
            raise err
        return cls(cfg.data['model_path'], cfg.data['decoder'], language=cfg.data['language'], cuda=cfg.data['cuda'])
