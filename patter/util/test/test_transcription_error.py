import unittest
from patter.util import TranscriptionError


class TestTranscriptionError(unittest.TestCase):
    def test_simple_error_calculation(self):
        hyp = "Ryan is cool"
        ref = "Brian is cool"

        err = TranscriptionError.calculate(hyp, ref)
        self.assertEqual(err.wer, 100 * 1/3, "Word Error Rate is incorrect")
        self.assertEqual(err.cer, 100 * 3/13, "Character Error Rate is incorrect")

    def test_combined_error_calculation(self):
        hyp1 = "Ryan is cool"
        ref1 = "Brian is cool"

        err1 = TranscriptionError.calculate(hyp1, ref1)
        self.assertEqual(err1.wer, 100 * 1/3, "Word Error Rate is incorrect")

        hyp2 = "A well-transcribed long example should get more wait"
        ref2 = "A well-transcribed long example should get more weight"

        err2 = TranscriptionError.calculate(hyp2, ref2)
        self.assertEqual(err2.wer, 100 * 1/8, "Word Error Rate is incorrect")

        err3 = err1 + err2
        self.assertEqual(err3.wer, 100 * 2/11, "Combined word error rate is incorrect")
        self.assertEqual(err1.wer, 100 * 1/3, "Word Error Rate is incorrect")
        self.assertEqual(err2.wer, 100 * 1/8, "Word Error Rate is incorrect")

    def test_inplace_addition_calculation(self):
        err = TranscriptionError()

        hyp1 = "ryan is cool"
        ref1 = "brian is cool"

        err += TranscriptionError.calculate(hyp1, ref1)
        self.assertEqual(err.wer, 100 * 1/3, "Word Error Rate is incorrect")

        hyp2 = "A well-transcribed long example should get more wait"
        ref2 = "A well-transcribed long example should get more weight"

        err += TranscriptionError.calculate(hyp2, ref2)
        self.assertEqual(err.wer, 100 * 2/11, "Combined word error rate is incorrect")
        self.assertEqual(err.cer, 100 * 5/67, "Combined character error rate is incorrect")
