"""
Data preprocessing utilities for Chinese-English machine translation
"""

import json
import re
import jieba
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Tuple
from config import Config


# nltk.download('punkt_tab', download_dir="./nltk_data")


class Preprocessor:
    """Handles data loading, cleaning, and tokenization"""

    def __init__(self):
        self.config = Config()

    def load_data(self, filepath: str) -> List[dict]:
        """Load JSONL data"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def clean_text(self, text: str, lang: str) -> str:
        """Clean text by removing illegal characters"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        return text

    def tokenize_chinese(self, text: str) -> List[str]:
        """Tokenize Chinese text using Jieba"""
        text = self.clean_text(text, 'zh')
        tokens = list(jieba.cut(text))
        return [token for token in tokens if token.strip()]

    def tokenize_english(self, text: str) -> List[str]:
        """Tokenize English text using simple word splitting"""
        text = self.clean_text(text, 'en')
        # Simple tokenization: split by space and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        # tokens = word_tokenize(text.lower())
        return tokens

    # def filter_by_length(self, src_tokens: List[str], tgt_tokens: List[str],
    #                      max_length: int = None) -> bool:
    #     """Check if sentence pair should be kept based on length"""
    #     if max_length is None:
    #         max_length = self.config.MAX_LENGTH

    #     if len(src_tokens) == 0 or len(tgt_tokens) == 0:
    #         return False

    #     if len(src_tokens) > max_length or len(tgt_tokens) > max_length:
    #         return False

    #     return True

    def truncate_by_length(self, src_tokens: List[str], tgt_tokens: List[str],
                         max_length: int = None) -> bool:
        """Check if sentence pair should be kept based on length"""
        if max_length is None:
            max_length = self.config.MAX_LENGTH

        if len(src_tokens) > max_length:
            src_tokens = src_tokens[:max_length]
        if len(tgt_tokens) > max_length:
            tgt_tokens = tgt_tokens[:max_length]

        return src_tokens, tgt_tokens

    def preprocess_data(self, data: List[dict], max_length: int = None, return_text: bool = True, en_key: str = "en", zh_key: str = "zh") -> List[Tuple[List[str], List[str]]]:
        """
        Preprocess dataset: tokenize and filter
        Returns list of (zh_tokens, en_tokens) tuples
        """
        preprocessed = []

        for item in data:
            zh_text = item[zh_key]
            en_text = item[en_key]

            if not return_text:
                # Tokenize
                zh_tokens = self.tokenize_chinese(zh_text)
                en_tokens = self.tokenize_english(en_text)
            else:
                zh_tokens = zh_text
                en_tokens = en_text

            # Filter by length
            # if self.filter_by_length(zh_tokens, en_tokens, max_length):
            #     preprocessed.append((zh_tokens, en_tokens))
            if len(zh_tokens) == 0 or len(en_tokens) == 0:
                continue
            # if len(zh_tokens) / len(en_tokens) > 4.0 or len(zh_tokens) / len(en_tokens) < 0.25:
            if len(zh_tokens) / len(en_tokens) > 6.0 or len(zh_tokens) / len(en_tokens) < 0.16:
                continue
            preprocessed.append(self.truncate_by_length(zh_tokens, en_tokens, max_length))

        # breakpoint()

        return preprocessed

    def load_and_preprocess(self, filepath: str, max_length: int = None, return_text: bool = True, zh_key: str = "zh", en_key: str = "en") -> List[Tuple[List[str], List[str]]]:
        """Load and preprocess data in one step"""
        data = self.load_data(filepath)
        # breakpoint()
        return self.preprocess_data(data, max_length, return_text, zh_key=zh_key, en_key=en_key)


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = Preprocessor()

    # Test on small training set
    print("Loading and preprocessing train_10k.jsonl...")
    data = preprocessor.load_and_preprocess(Config.TRAIN_SMALL_PATH)
    print(f"Loaded {len(data)} sentence pairs")

    # Show examples
    print("\nFirst 3 examples:")
    for i, (zh, en) in enumerate(data[:3]):
        print(f"\nExample {i+1}:")
        print(f"ZH: {' '.join(zh)}")
        print(f"EN: {' '.join(en)}")
