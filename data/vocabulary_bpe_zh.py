"""
Vocabulary implementation using Byte Pair Encoding (BPE) tokenizer
"""

from typing import List, Optional, Union
import torch
import pickle
import json
import os
import re
from config import Config
from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, Regex
from tokenizers.models import BPE


class BPEVocabularyZH:
    """
    Vocabulary class using Hugging Face's tokenizers library for BPE tokenization
    """

    def __init__(self, name: str = "bpe_vocab"):
        """
        Initialize BPE Vocabulary

        Args:
            name: Name of the vocabulary
        """
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
        except ImportError:
            raise ImportError(
                "tokenizers library is required. Install with: pip install tokenizers"
            )

        self.name = name
        self.language = "zh"
        self.config = Config()

        # Special tokens
        self.pad_token = self.config.PAD_TOKEN
        self.unk_token = self.config.UNK_TOKEN
        self.bos_token = self.config.BOS_TOKEN
        self.eos_token = self.config.EOS_TOKEN

        # Tokenizer
        self.tokenizer = None
        self.is_trained = False

        # # Initialize special tokens
        # self._init_special_tokens()

    def _initialize_tokenizer(self):
        """Initialize BPE tokenizer if not already initialized"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))

    def train_from_texts(self,
                         texts: List[str],
                         max_size: int = 30000,
                         min_freq: int = 2,
                         special_tokens: Optional[List[str]] = None):
        """
        Train BPE tokenizer from list of texts

        Args:
            texts: List of text strings
            max_size: Size of the vocabulary
            min_freq: Minimum frequency for a subword to be included
            special_tokens: List of special tokens
        """
        from tokenizers.trainers import BpeTrainer

        if special_tokens is None:
            special_tokens = [
                self.pad_token, self.bos_token, self.eos_token, self.unk_token
            ]

        # Initialize BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))

        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(), # 如果你需要不区分大小写
            normalizers.Strip(),     # 自动去除首尾空格
            normalizers.Replace(Regex(r"\s+"), "") # 【核心】qudiao空白
        ])

        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            # pre_tokenizers.Metaspace(replacement=" "),
            # pre_tokenizers.Whitespace(),  # Split by space
            pre_tokenizers.Punctuation(),  # Split punctuation off words
            pre_tokenizers.Digits(individual_digits=True)  # split digits
        ])
        # self.tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=" ")

        # This tells the tokenizer how to glue tokens back together (join with spaces)
        self.tokenizer.decoder = decoders.BPEDecoder()

        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=max_size,
            min_frequency=min_freq,
            special_tokens=special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
            # continuing_subword_prefix="##"
        )

        # Train tokenizer
        print(f"Training BPE tokenizer on {len(texts)} texts...")
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        self.is_trained = True
        print(f"BPE vocabulary trained with {self.tokenizer.get_vocab_size()} tokens")

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
            raise ValueError("Tokenizer must be trained before encoding")

        indices = []

        if add_bos:
            indices.append(self.bos_idx)

        for token in tokens:
            # todo: tokenizer won't add special tokens here
            indices.append(self.tokenizer.encode(token, add_special_tokens=add_bos or add_eos).ids[0])
        # breakpoint()

        if add_eos:
            indices.append(self.eos_idx)

        return indices

        # encoding = self.tokenizer.encode(text, add_special_tokens=add_bos or add_eos)

        # # Manually handle BOS/EOS if needed
        # tokens = []
        # if add_bos:
        #     tokens.append(self.bos_idx)

        # tokens.extend(encoding.ids)

        # if add_eos:
        #     tokens.append(self.eos_idx)

        # return tokens
        

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding")

        # Get special token IDs
        special_ids = {self.pad_idx, self.bos_idx, self.eos_idx}

        # Filter out special tokens if requested
        if skip_special:
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        # breakpoint()

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """
        Encode a batch of texts

        Args:
            texts: List of input text strings
            max_length: Maximum sequence length

        Returns:
            List of token ID lists
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")

        encodings = self.tokenizer.encode_batch(
            texts,
            max_length=max_length,
            padding=True if max_length else "max_length",
            truncation=True if max_length else False,
            add_special_tokens=True
        )

        return [enc.ids for enc in encodings]

    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """
        Decode a batch of token IDs

        Args:
            batch_ids: List of token ID lists

        Returns:
            List of decoded text strings
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding")

        return self.tokenizer.decode_batch(batch_ids, skip_special_tokens=True)

    def segement_text(self, texts: List[str], max_length: Optional[int] = None, is_pretokenized: bool = False, output_ids: bool = True, add_bos: bool = True, add_eos: bool = True) -> List[List[str]]:
        """
        Segment texts into subword tokens

        Args:
            texts: List of input text strings
            max_length: Maximum sequence length

        Returns:
            List of subword token lists
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before segmentation")

        # print(type(self.tokenizer))
        encodings = self.tokenizer.encode_batch(
            texts,
            add_special_tokens=True,
            is_pretokenized=is_pretokenized,
        )

        if output_ids:
            tokens = [enc.ids for enc in encodings]
        else:
            tokens = [enc.tokens for enc in encodings]

        for i, token in enumerate(tokens):
            if max_length and len(token) > max_length:
                tokens[i] = token[:max_length]

            if output_ids:
                if add_bos:
                    tokens[i] = [self.bos_idx] + tokens[i]
                if add_eos:
                    tokens[i] = tokens[i] + [self.eos_idx]
                # breakpoint()
            else: 
                if add_bos:
                    tokens[i] = [self.bos_token] + tokens[i]
                if add_eos:
                    tokens[i] = tokens[i] + [self.eos_token]

        return tokens

    def ids_to_tokens(self, ids: Union[List[int], torch.Tensor], skip_special: bool = True) -> List[str]:
        """
        Converts a list of IDs (or a Tensor) into their raw BPE token strings.
        
        Args:
            tokenizer: The trained HuggingFace tokenizer object.
            ids: A list of integers or a 1D PyTorch Tensor containing token IDs.
            
        Returns:
            A list of strings (e.g., ['<bos>', ' Re', 'cor', 'ds', ...])
        """
        
        # 1. Handle PyTorch Tensors (convert to standard Python list)
        if torch.is_tensor(ids):
            ids = ids.tolist()

        special_indices = {
            self.pad_idx,
            self.bos_idx,
            self.eos_idx
        }
            
        # 2. Convert each ID to its Token string
        # id_to_token returns None if ID is out of vocabulary, so we handle that safely
        tokens = []
        for i in ids:
            token = self.tokenizer.id_to_token(i)
            if skip_special and i in special_indices:
                continue
            if token is None:
                tokens.append(self.unk_token)
            else:
                tokens.append(token)
                
        return tokens

    def save(self, tokenizer_filepath: str, vocab_filepath: str):
        """
        Save trained tokenizer and vocabulary to file

        Args:
            tokenizer_filepath: Path to save the trained tokenizer
            vocab_filepath: Path to save the vocabulary
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before saving")

        os.makedirs(os.path.dirname(tokenizer_filepath), exist_ok=True)
        self.tokenizer.save(tokenizer_filepath)
        print(f"BPE tokenizer saved to {tokenizer_filepath}")

        os.makedirs(os.path.dirname(vocab_filepath), exist_ok=True)
        with open(vocab_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False, indent=4)
        print(f"BPE vocabulary saved to {vocab_filepath}")

    def load(self, filepath: str):
        """
        Load trained tokenizer from file

        Args:
            filepath: Path to the saved tokenizer
        """
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(filepath)
        self.is_trained = True
        print(f"BPE tokenizer loaded from {filepath} (size: {self.tokenizer.get_vocab_size()})")

    def get_vocab(self) -> dict:
        """
        Get vocabulary as dictionary

        Returns:
            Dictionary mapping tokens to IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before getting vocab")

        return self.tokenizer.get_vocab()

    def __len__(self) -> int:
        """Get vocabulary size"""
        if not self.is_trained:
            return 0
        return self.tokenizer.get_vocab_size()

    @property
    def pad_idx(self) -> int:
        """Get pad token ID"""
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def unk_idx(self) -> int:
        """Get unknown token ID"""
        return self.tokenizer.token_to_id(self.unk_token)

    @property
    def bos_idx(self) -> int:
        """Get beginning-of-sentence token ID"""
        return self.tokenizer.token_to_id(self.bos_token)

    @property
    def eos_idx(self) -> int:
        """Get end-of-sentence token ID"""
        return self.tokenizer.token_to_id(self.eos_token)

if __name__ == "__main__":
    # Example usage
    import json

    # Test with a small dataset
    corpus_path = Config.TRAIN_SMALL_PATH

    # Load sample texts
    texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Use first 100 lines for testing
                break
            data = json.loads(line)
            texts.append(data['en'])

    # Train English BPE vocabulary
    print("Training English BPE vocabulary...")
    en_vocab = BPEVocabulary("English_BPE")
    en_vocab.train_from_texts(texts, vocab_size=5000, min_frequency=2)

    # Test encoding/decoding
    test_text = "Hello world! This is a test sentence."
    print(f"\nOriginal text: {test_text}")

    encoded = en_vocab.encode(test_text, add_bos=True, add_eos=True)
    print(f"Encoded: {encoded}")

    decoded = en_vocab.decode(encoded)
    print(f"Decoded: {decoded}")

    # Save tokenizer
    en_vocab.save("./data/en_bpe_tokenizer.json")

    # Load tokenizer
    print("\nLoading tokenizer from file...")
    loaded_vocab = BPEVocabulary("Loaded_BPE")
    loaded_vocab.load("./data/en_bpe_tokenizer.json")

    # Test loaded tokenizer
    test_text2 = "This is another test."
    encoded2 = loaded_vocab.encode(test_text2)
    decoded2 = loaded_vocab.decode(encoded2)
    print(f"Test with loaded tokenizer: {test_text2} -> {decoded2}")
