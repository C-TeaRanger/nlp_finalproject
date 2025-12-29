"""
PyTorch DataLoader for machine translation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from config import Config


class TranslationDataset(Dataset):
    """Dataset for Chinese-English translation"""

    def __init__(self, data: List[Tuple[List[str], List[str]]], src_vocab, tgt_vocab, inverse: bool = False):
        """
        Args:
            data: List of (src_tokens, tgt_tokens) tuples
            src_vocab: Source language vocabulary
            tgt_vocab: Target language vocabulary
        """
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.inverse = inverse
        # breakpoint()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.data[idx]
        # src_tokens, tgt_tokens = self.data[100]

        # Encode tokens to indices
        # src_indices = self.src_vocab.encode(src_tokens, add_bos=True, add_eos=True)
        # tgt_indices = self.tgt_vocab.encode(tgt_tokens, add_bos=True, add_eos=True)

        # breakpoint()
        if self.inverse:
            # 进行句子反转处理
             # 1. 取出中间的内容 (去掉第1个和最后1个)
            content = src_tokens[1:-1]
            
            # 2. 反转中间内容
            reversed_content = content[::-1]
            
            # 3. 重新拼接：BOS + 反转后的中间 + EOS
            # 注意：这里假设 src_tokens[0] 是 BOS， src_tokens[-1] 是 EOS
            src_tokens = [src_tokens[0]] + reversed_content + [src_tokens[-1]]

        # # 1. 取出中间的内容 (去掉第1个和最后1个)
        # content = src_tokens[1:-1]
        
        # # 2. 反转中间内容
        # reversed_content = content[::-1]
        
        # # 3. 重新拼接：BOS + 反转后的中间 + EOS
        # # 注意：这里假设 src_tokens[0] 是 src_tokens[-1] 是 EOS
        # src_tokens = [src_tokens[0]] + reversed_content + [src_tokens[-1]]

        # return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)


def collate_fn(batch, pad_idx):
    """
    Collate function to pad sequences in a batch

    Args:
        batch: List of (src, tgt) tensors
        pad_idx: Padding index

    Returns:
        src_batch: (batch_size, max_src_len)
        tgt_batch: (batch_size, max_tgt_len)
        src_lengths: (batch_size,)
        tgt_lengths: (batch_size,)
    """
    src_batch, tgt_batch = zip(*batch)

    # Get lengths before padding
    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch], dtype=torch.long)

    # Pad sequences
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    return src_batch, tgt_batch, src_lengths, tgt_lengths


def get_dataloader(data: List[Tuple[List[str], List[str]]],
                   src_vocab,
                   tgt_vocab,
                   batch_size: int,
                   shuffle: bool = True,
                   num_workers: int = 0,
                   sampler: torch.utils.data.Sampler = None):
    """
    Create DataLoader for translation dataset

    Args:
        data: List of (src_tokens, tgt_tokens) tuples
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading

    Returns:
        DataLoader
    """
    dataset = TranslationDataset(data, src_vocab, tgt_vocab)

    # Create collate function with pad_idx
    def collate_wrapper(batch):
        return collate_fn(batch, tgt_vocab.pad_idx)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_wrapper,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader


if __name__ == "__main__":
    # Test dataloader
    from preprocessor import Preprocessor
    from vocabulary import Vocabulary

    config = Config()
    preprocessor = Preprocessor()

    # Load and preprocess data
    print("Loading data...")
    data = preprocessor.load_and_preprocess(config.TRAIN_SMALL_PATH)
    print(f"Loaded {len(data)} samples")

    # Build vocabularies
    zh_tokens = [zh for zh, en in data]
    en_tokens = [en for zh, en in data]

    zh_vocab = Vocabulary("Chinese")
    zh_vocab.build_vocab(zh_tokens)

    en_vocab = Vocabulary("English")
    en_vocab.build_vocab(en_tokens)

    # Create dataloader
    dataloader = get_dataloader(data, zh_vocab, en_vocab, batch_size=4, shuffle=True)

    # Test batch
    print("\nTesting dataloader:")
    for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Source shape: {src.shape}")
        print(f"  Target shape: {tgt.shape}")
        print(f"  Source lengths: {src_len}")
        print(f"  Target lengths: {tgt_len}")

        # Decode first sample
        print(f"\n  First sample:")
        print(f"    SRC: {' '.join(zh_vocab.decode(src[0].tolist()))}")
        print(f"    TGT: {' '.join(en_vocab.decode(tgt[0].tolist()))}")

        if batch_idx >= 2:
            break
