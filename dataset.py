"""
dataset.py — Multi30k Dataset with spaCy tokenization
DA6401 Assignment 3
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class Vocabulary:
    """Simple token↔index vocabulary."""

    PAD_IDX = 1
    UNK_IDX = 0
    SOS_IDX = 2
    EOS_IDX = 3

    SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]

    def __init__(self):
        self.itos = {}   # index → token
        self.stoi = {}   # token → index

    def build(self, token_lists, min_freq: int = 2):
        counter = Counter(tok for tokens in token_lists for tok in tokens)
        self.itos = {i: s for i, s in enumerate(self.SPECIALS)}
        self.stoi = {s: i for i, s in self.itos.items()}
        for tok, freq in sorted(counter.items()):
            if freq >= min_freq and tok not in self.stoi:
                idx = len(self.itos)
                self.itos[idx] = tok
                self.stoi[tok] = idx

    def __len__(self):
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, "<unk>")

    def lookup_indices(self, tokens):
        return [self.stoi.get(t, self.UNK_IDX) for t in tokens]


class Multi30kDataset(Dataset):
    """
    Multi30k German→English dataset.
    Wraps HuggingFace bentrevett/multi30k with spaCy tokenization.
    """

    def __init__(self, split: str = "train", src_vocab=None, tgt_vocab=None, min_freq: int = 2):
        from datasets import load_dataset
        import spacy

        self.split = split
        self.nlp_de = spacy.load("de_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm")

        raw = load_dataset("bentrevett/multi30k", split=split)
        self.src_raw = [ex["de"] for ex in raw]
        self.tgt_raw = [ex["en"] for ex in raw]

        self.src_tokens = [self._tokenize_de(s) for s in self.src_raw]
        self.tgt_tokens = [self._tokenize_en(s) for s in self.tgt_raw]

        if src_vocab is None:
            self.src_vocab = Vocabulary()
            self.src_vocab.build(self.src_tokens, min_freq=min_freq)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = Vocabulary()
            self.tgt_vocab.build(self.tgt_tokens, min_freq=min_freq)
        else:
            self.tgt_vocab = tgt_vocab

        self.src_data, self.tgt_data = self.process_data()

    def _tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self.nlp_de.tokenizer(text)]

    def _tokenize_en(self, text: str):
        return [tok.text.lower() for tok in self.nlp_en.tokenizer(text)]

    def build_vocab(self):
        src_v = Vocabulary()
        src_v.build(self.src_tokens)
        tgt_v = Vocabulary()
        tgt_v.build(self.tgt_tokens)
        return src_v, tgt_v

    def process_data(self):
        src_data, tgt_data = [], []
        sos, eos = Vocabulary.SOS_IDX, Vocabulary.EOS_IDX
        for src_toks, tgt_toks in zip(self.src_tokens, self.tgt_tokens):
            src_ids = [sos] + self.src_vocab.lookup_indices(src_toks) + [eos]
            tgt_ids = [sos] + self.tgt_vocab.lookup_indices(tgt_toks) + [eos]
            src_data.append(src_ids)
            tgt_data.append(tgt_ids)
        return src_data, tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), \
               torch.tensor(self.tgt_data[idx], dtype=torch.long)


def collate_fn(batch, pad_idx: int = Vocabulary.PAD_IDX):
    """Pad variable-length sequences in a batch."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def build_dataloaders(batch_size: int = 128, num_workers: int = 0):
    """Build train / val / test DataLoaders and return vocabs."""
    train_ds = Multi30kDataset("train")
    val_ds   = Multi30kDataset("validation", src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)
    test_ds  = Multi30kDataset("test",       src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab)

    fn = lambda b: collate_fn(b)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=fn)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=num_workers, collate_fn=fn)

    return train_loader, val_loader, test_loader, train_ds.src_vocab, train_ds.tgt_vocab
