"""一个从零实现的简单分词器：中文按单字切分，英文按单词，数字按整串。"""
import re
from collections import Counter

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+|[^\s\w]")
PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.stoi = {tok: i for i, tok in enumerate(vocab)}

    @classmethod
    def build(cls, texts, max_size=20000, min_freq=2):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        words = [w for w, c in counter.most_common(max_size) if c >= min_freq]
        return cls([PAD, UNK, CLS] + words)

    def encode(self, text, max_len):
        """返回 (token id 列表, 注意力掩码)，开头固定放一个 <cls>，用它的输出做分类。"""
        unk = self.stoi[UNK]
        ids = [self.stoi[CLS]] + [self.stoi.get(t, unk) for t in tokenize(text)]
        ids = ids[:max_len]
        mask = [1] * len(ids)
        pad = max_len - len(ids)
        return ids + [self.stoi[PAD]] * pad, mask + [0] * pad
