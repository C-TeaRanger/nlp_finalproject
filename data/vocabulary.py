"""
Vocabulary construction and management
"""

from collections import Counter
from typing import List, Tuple
import pickle
from config import Config


class Vocabulary:
    """Vocabulary class for managing word-to-index mappings"""

    def __init__(self, name: str = "vocab"):
        self.name = name
        self.config = Config()

        # Special tokens
        self.pad_token = self.config.PAD_TOKEN
        self.unk_token = self.config.UNK_TOKEN
        self.bos_token = self.config.BOS_TOKEN
        self.eos_token = self.config.EOS_TOKEN

        # Mappings
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # Initialize with special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize special tokens"""
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build_vocab(self, tokenized_data: List[List[str]], min_freq: int = None, max_size: int = None):
        """Build vocabulary from tokenized data"""
        if min_freq is None:
            min_freq = self.config.MIN_FREQ
        if max_size is None:
            max_size = self.config.MAX_VOCAB_SIZE

        # Count word frequencies
        for tokens in tokenized_data:
            self.word_freq.update(tokens)

        # Sort by frequency
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        # Add words to vocabulary
        for word, freq in sorted_words:
            if freq < min_freq:
                break
            if len(self.word2idx) >= max_size:
                break

            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"{self.name} vocabulary size: {len(self.word2idx)}")
        print(f"  - Total words in corpus: {len(self.word_freq)}")
        print(f"  - Words with freq >= {min_freq}: {sum(1 for f in self.word_freq.values() if f >= min_freq)}")

    def encode(self, tokens: List[str], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert tokens to indices"""
        indices = []

        if add_bos:
            indices.append(self.word2idx[self.bos_token])

        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx[self.unk_token]))

        if add_eos:
            indices.append(self.word2idx[self.eos_token])

        return indices

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """Convert indices to tokens"""
        special_indices = {
            self.word2idx[self.pad_token],
            self.word2idx[self.bos_token],
            self.word2idx[self.eos_token]
        }

        tokens = []
        for idx in indices:
            if skip_special and idx in special_indices:
                continue
            tokens.append(self.idx2word.get(idx, self.unk_token))

        return tokens

    def __len__(self):
        return len(self.word2idx)

    def save(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'name': self.name,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")

    def load(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)

        self.name = vocab_data['name']
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.word_freq = vocab_data['word_freq']
        print(f"Vocabulary loaded from {filepath} (size: {len(self.word2idx)})")

    @property
    def pad_idx(self):
        return self.word2idx[self.pad_token]

    @property
    def unk_idx(self):
        return self.word2idx[self.unk_token]

    @property
    def bos_idx(self):
        return self.word2idx[self.bos_token]

    @property
    def eos_idx(self):
        return self.word2idx[self.eos_token]


if __name__ == "__main__":
    # Test vocabulary building
    from preprocessor import Preprocessor

    preprocessor = Preprocessor()
    data = preprocessor.load_and_preprocess(Config.TRAIN_SMALL_PATH)

    # Separate Chinese and English
    zh_tokens = [zh for zh, en in data]
    en_tokens = [en for zh, en in data]

    # Build vocabularies
    zh_vocab = Vocabulary("Chinese")
    zh_vocab.build_vocab(zh_tokens)

    en_vocab = Vocabulary("English")
    en_vocab.build_vocab(en_tokens)

    # Test encoding/decoding
    print("\nTest encoding/decoding:")
    sample_zh = zh_tokens[0]
    print(f"Original: {' '.join(sample_zh)}")
    encoded = zh_vocab.encode(sample_zh, add_bos=True, add_eos=True)
    print(f"Encoded: {encoded}")
    decoded = zh_vocab.decode(encoded)
    print(f"Decoded: {' '.join(decoded)}")
