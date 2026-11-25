"""CPU-only corpus helpers, shared by data preparation and training."""
import hashlib

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD, BOS, EOS, MASK, SEP, EN, RU, TASK = range(256, 264)
CONTROL = {"en": EN, "ru": RU}
DOMAINS = ("simple", "education", "math", "python", "logic", "reasoning")
QUOTAS = (.50, .20, .15, .05, .05, .05)

BPE_SPECIAL = ["<pad>", "<bos>", "<eos>", "<en>", "<ru>", "<task>", "<sep>"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for frame in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(frame)
    return digest.hexdigest()


def train_tokenizer(texts, path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=16384, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=BPE_SPECIAL, show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(str(path))
    return tokenizer


class Codec:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.pad = PAD if tokenizer is None else tokenizer.token_to_id("<pad>")
        self.bos = BOS if tokenizer is None else tokenizer.token_to_id("<bos>")
        self.eos = EOS if tokenizer is None else tokenizer.token_to_id("<eos>")
        self.task = TASK if tokenizer is None else tokenizer.token_to_id("<task>")

    def lang(self, language):
        return CONTROL[language] if self.tokenizer is None else self.tokenizer.token_to_id(f"<{language}>")

    def content(self, text):
        text.encode("utf-8", errors="strict")
        if self.tokenizer is None:
            return list(text.encode("utf-8"))
        # Disable added control-token recognition in content by encoding ordinary
        # spans separated by a literal control spelling as byte alphabet symbols.
        import re
        if not any(marker in text for marker in BPE_SPECIAL):
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        split = "(" + "|".join(re.escape(marker) for marker in BPE_SPECIAL) + ")"
        ids = []
        for part in re.split(split, text):
            if part in BPE_SPECIAL:
                for char in part:
                    ids.extend(self.tokenizer.encode(char, add_special_tokens=False).ids)
            elif part:
                ids.extend(self.tokenizer.encode(part, add_special_tokens=False).ids)
        return ids

    def decode(self, ids):
        if self.tokenizer is not None:
            return self.tokenizer.decode(ids)
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def ids(self, text, language, eos=True, task=False):
        return [self.bos, self.lang(language)] + ([self.task] if task else []) + self.content(text) + (
            [self.eos] if eos else [])
