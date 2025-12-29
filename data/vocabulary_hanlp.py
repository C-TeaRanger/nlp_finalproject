"""
Vocabulary implementation using HanLP tokenizer
"""

from typing import List, Optional, Dict, Set
import pickle
import json
import os
from config import Config
from collections import Counter


class HanLPVocabulary:
    """
    Vocabulary class using HanLP tokenizer for Chinese text processing
    """

    def __init__(self, name: str = "hanlp_vocab", language: str = "zh"):
        """
        Initialize HanLP Vocabulary

        Args:
            name: Name of the vocabulary
            language: Language code ('zh' for Chinese, 'en' for English)
        """
        try:
            import hanlp
        except ImportError:
            raise ImportError(
                "hanlp library is required. Install with: pip install hanlp"
            )

        self.name = name
        self.language = language
        self.config = Config()

        # Special tokens
        self.pad_token = self.config.PAD_TOKEN
        self.unk_token = self.config.UNK_TOKEN
        self.bos_token = self.config.BOS_TOKEN
        self.eos_token = self.config.EOS_TOKEN

        # Vocabulary mappings
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()

        # Tokenizer
        self.tokenizer = None
        self.is_trained = False

        # Initialize special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def _initialize_tokenizer(self):
        """Initialize the appropriate HanLP tokenizer based on language"""
        import hanlp

        if self.language == "zh":
            # For Chinese: Use Chinese Word Segmentation (CWS) tokenizer
            # Using a pre-trained TokFinisher model
            self.tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
        elif self.language == "en":
            # For English: Use English tokenizer
            # Using a simple PTB tokenizer or treebank tokenizer
            self.tokenizer = hanlp.load('CONVSEG_ENGLISH_TOK')
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def train(self,
              tokenized_data: List[List[str]],
              min_freq: int = None,
              max_size: Optional[int] = None):
        """
        Build vocabulary from tokenized data

        Args:
            tokenized_data: List of tokenized sentences (each sentence is a list of tokens)
            min_freq: Minimum frequency for a token to be included
            max_size: Maximum vocabulary size
        """
        if min_freq is None:
            min_freq = self.config.MIN_FREQ
        if max_size is None:
            max_size = self.config.MAX_VOCAB_SIZE

        # Count word frequencies
        for tokens in tokenized_data:
            self.word_freq.update(tokens)

        # # Add words to vocabulary (skip special tokens which are already added)
        # existing_special_count = len([t for t in self.word2idx.keys()
        #                              if t in [self.pad_token, self.unk_token,
        #                                      self.bos_token, self.eos_token]])

        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        for word, freq in sorted_words:
            if freq < min_freq:
                break
            if len(self.word2idx) >= max_size:
                break

            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.is_trained = True
        print(f"{self.name} vocabulary size: {len(self.word2idx)}")
        print(f"  - Total unique words in corpus: {len(self.word_freq)}")
        print(f"  - Words with freq >= {min_freq}: {sum(1 for f in self.word_freq.values() if f >= min_freq)}")

    def train_from_texts(self,
                         texts: List[str],
                         min_freq: int = 2,
                         max_size: Optional[int] = None):
        """
        Build vocabulary from raw texts using HanLP tokenizer

        Args:
            texts: List of raw text strings
            min_freq: Minimum frequency for a token to be included
            max_size: Maximum vocabulary size
        """
        if self.tokenizer is None:
            self._initialize_tokenizer()

        # Tokenize all texts
        print(f"Tokenizing {len(texts)} texts using HanLP...")
        tokenized_data = []

        try:
            from tqdm import tqdm
            for text in tqdm(texts, desc="Tokenizing"):
                tokens = self.tokenizer(text)
                if isinstance(tokens, dict):
                    # HanLP might return a dict for some models
                    tokens = tokens.get('tok', [])
                tokenized_data.append(tokens)
        except ImportError:
            # Fallback without tqdm
            for text in texts:
                tokens = self.tokenizer(text)
                if isinstance(tokens, dict):
                    tokens = tokens.get('tok', [])
                tokenized_data.append(tokens)

        # Build vocabulary from tokenized data
        self.train(tokenized_data, min_freq=min_freq, max_size=max_size)

    def encode(self, tokens: List[str], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text string
            add_bos: Whether to add beginning-of-sentence token
            add_eos: Whether to add end-of-sentence token

        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise ValueError("Vocabulary must be trained before encoding")

        if self.tokenizer is None:
            self._initialize_tokenizer()

        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")

        indices = []

        if add_bos:
            indices.append(self.bos_idx)

        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx[self.unk_token]))

        if add_eos:
            indices.append(self.eos_idx)
        
        # breakpoint()

        return indices
        
        # tokens = self.tokenizer(text)

        # # Convert tokens to indices
        # indices = []

        # if add_bos:
        #     indices.append(self.word2idx[self.bos_token])

        # for token in tokens:
        #     indices.append(self.word2idx.get(token, self.word2idx[self.unk_token]))

        # if add_eos:
        #     indices.append(self.word2idx[self.eos_token])

        # return indices

    # def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
    #     """
    #     Decode token IDs to tokens

    #     Args:
    #         indices: List of token IDs
    #         skip_special: Whether to skip special tokens

    #     Returns:
    #         List of tokens (not joined into text)
    #     """
    #     if not self.is_trained:
    #         raise ValueError("Vocabulary must be trained before decoding")

    #     # Get special token IDs
    #     special_indices = {
    #         self.word2idx[self.pad_token],
    #         self.word2idx[self.bos_token],
    #         self.word2idx[self.eos_token]
    #     }

    #     tokens = []
    #     for tid in token_ids:
    #         if skip_special and tid in special_ids:
    #             continue
    #         tokens.append(self.idx2word.get(tid, self.unk_token))

    #     return tokens

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """
        Encode a batch of texts

        Args:
            texts: List of input text strings
            max_length: Maximum sequence length (truncates if longer)

        Returns:
            List of token ID lists
        """
        if not self.is_trained:
            raise ValueError("Vocabulary must be trained before encoding")

        encoded_batch = []
        for text in texts:
            encoded = self.encode(text)
            if max_length and len(encoded) > max_length:
                encoded = encoded[:max_length]
            encoded_batch.append(encoded)

        return encoded_batch

    def decode_batch(self, batch_ids: List[List[int]]) -> List[List[str]]:
        """
        Decode a batch of token IDs

        Args:
            batch_ids: List of token ID lists

        Returns:
            List of token lists
        """
        return [self.decode(ids) for ids in batch_ids]

    def segement_text(self, texts: List[str], max_length: Optional[int] = None) -> List[List[str]]:
        """
        Segment a batch of texts into tokens using HanLP tokenizer

        Args:
            texts: List of input text strings
            max_length: Maximum sequence length (truncates if longer)

        Returns:
            List of token lists
        """
        if self.tokenizer is None:
            self._initialize_tokenizer()

        segmented_batch = []
        for text in texts:
            tokens = self.tokenizer(text)
            if isinstance(tokens, dict):
                tokens = tokens.get('tok', [])
            # breakpoint()
            if max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            segmented_batch.append(tokens)

        return segmented_batch

    def save(self, filepath: str):
        """
        Save vocabulary to file

        Args:
            filepath: Path to save the vocabulary
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        vocab_data = {
            'name': self.name,
            'language': self.language,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq,
            'is_trained': self.is_trained
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)

        print(f"Vocabulary saved to {filepath} (size: {len(self.word2idx)})")

    def load(self, filepath: str):
        """
        Load vocabulary from file

        Args:
            filepath: Path to the saved vocabulary
        """
        with open(filepath, 'rb') as f:
            vocab_data = json.load(f)

        self.name = vocab_data['name']
        self.language = vocab_data['language']
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.word_freq = vocab_data['word_freq']
        self.is_trained = vocab_data['is_trained']

        # Initialize tokenizer
        self._initialize_tokenizer()

        print(f"Vocabulary loaded from {filepath} (size: {len(self.word2idx)})")

    def __len__(self) -> int:
        """Get vocabulary size"""
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        """Get pad token ID"""
        return self.word2idx[self.pad_token]

    @property
    def unk_idx(self) -> int:
        """Get unknown token ID"""
        return self.word2idx[self.unk_token]

    @property
    def bos_idx(self) -> int:
        """Get beginning-of-sentence token ID"""
        return self.word2idx[self.bos_token]

    @property
    def eos_idx(self) -> int:
        """Get end-of-sentence token ID"""
        return self.word2idx[self.eos_token]


if __name__ == "__main__":
    # Example usage
    import json

    # Test with Chinese text
    corpus_path = Config.TRAIN_SMALL_PATH

    # Load sample Chinese texts
    texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Use first 100 lines for testing
                break
            data = json.loads(line)
            texts.append(data['zh'])

    # Train Chinese HanLP vocabulary
    print("Training Chinese HanLP vocabulary...")
    zh_vocab = HanLPVocabulary("Chinese_HanLP", language="zh")
    zh_vocab.train_from_texts(texts, min_freq=2, max_size=5000)

    # Test encoding/decoding
    test_text = "你好世界！这是一个测试句子。"
    print(f"\nOriginal text: {test_text}")

    encoded = zh_vocab.encode(test_text, add_bos=True, add_eos=True)
    print(f"Encoded: {encoded}")

    decoded = zh_vocab.decode(encoded)
    print(f"Decoded tokens: {decoded}")

    decoded_text = zh_vocab.decode_to_text(encoded)
    print(f"Decoded text: {decoded_text}")

    # Save vocabulary
    zh_vocab.save("./data/zh_hanlp_vocab.json")

    # Load vocabulary
    print("\nLoading vocabulary from file...")
    loaded_vocab = HanLPVocabulary("Loaded_HanLP", language="zh")
    loaded_vocab.load("./data/zh_hanlp_vocab.json")

    # Test loaded vocabulary
    test_text2 = "这是另一个测试。"
    encoded2 = loaded_vocab.encode(test_text2)
    decoded2 = loaded_vocab.decode_to_text(encoded2)
    print(f"Test with loaded vocabulary: {test_text2} -> {decoded2}")
