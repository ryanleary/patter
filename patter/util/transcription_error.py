import Levenshtein as Lev


class TranscriptionError(object):
    def __init__(self, word_errors=None, char_errors=None, tokens=0, chars=0):
        self.word_errors = word_errors if word_errors is not None else {"delete": 0, "insert": 0, "replace": 0}
        self.char_errors = char_errors if char_errors is not None else {"delete": 0, "insert": 0, "replace": 0}
        self.tokens = tokens
        self.chars = chars

    @property
    def wer(self):
        return sum(self.word_errors.values()) / max(1, self.tokens)

    @property
    def cer(self):
        return sum(self.char_errors.values()) / max(1, self.chars)

    @classmethod
    def calculate(cls, hypothesis, reference):
        word_errors = TranscriptionError._get_word_errors(hypothesis, reference)
        char_errors = TranscriptionError._get_char_errors(hypothesis, reference)

        return cls(word_errors, char_errors, len(reference.split()), len(reference))

    @staticmethod
    def _get_word_errors(s1, s2):
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        ops = Lev.editops(''.join(w1), ''.join(w2))
        errors = {"delete": 0, "insert": 0, "replace": 0}
        for x in ops:
            errors[x[0]] += 1
        return errors

    @staticmethod
    def _get_char_errors(s1, s2):
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        ops = Lev.editops(s1, s2)
        errors = {"delete": 0, "insert": 0, "replace": 0}
        for x in ops:
            errors[x[0]] += 1
        return errors

    def __add__(self, other):
        word_errors = {"delete": 0, "insert": 0, "replace": 0}
        char_errors = {"delete": 0, "insert": 0, "replace": 0}
        for k in word_errors.keys():
            word_errors[k] = self.word_errors[k] + other.word_errors[k]
        for k in char_errors.keys():
            char_errors[k] = self.char_errors[k] + other.char_errors[k]
        tokens = self.tokens + other.tokens
        chars = self.chars + other.chars
        return TranscriptionError(word_errors, char_errors, tokens, chars)

    def __iadd__(self, other):
        for k in self.word_errors.keys():
            self.word_errors[k] = self.word_errors[k] + other.word_errors[k]
        for k in self.char_errors.keys():
            self.char_errors[k] = self.char_errors[k] + other.char_errors[k]
        self.tokens += other.tokens
        self.chars += other.chars
        return self
