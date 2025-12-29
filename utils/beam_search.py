"""
Beam search decoding utilities
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple
from config import Config
from .plot import plot_attention


class BeamSearchNode:
    """Node in beam search tree"""

    def __init__(self, token_id: int, log_prob: float, hidden_state, prev_node=None):
        self.token_id = token_id
        self.log_prob = log_prob
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.length = 1 if prev_node is None else prev_node.length + 1

    def get_sequence(self) -> List[int]:
        """Get sequence from root to current node"""
        sequence = []
        node = self
        while node is not None:
            sequence.append(node.token_id)
            node = node.prev_node
        return sequence[::-1]

    def get_avg_log_prob(self) -> float:
        """Get average log probability (normalized by length)"""
        return self.log_prob / self.length


def greedy_decode(model, src, src_len, tgt_vocab, max_length: int, device: str) -> List[int]:
    """
    Greedy decoding

    Args:
        model: Translation model
        src: Source sequence (1, src_len)
        src_len: Source length
        tgt_vocab: Target vocabulary
        max_length: Maximum decoding length
        device: Device

    Returns:
        Decoded sequence (list of token ids)
    """
    model.eval()

    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        actual_model = model.module
    else:
        actual_model = model

    with torch.no_grad():
        # Start with BOS token
        tgt = torch.tensor([[tgt_vocab.bos_idx]], dtype=torch.long, device=device)
        decoded = []

        # Encode source (for transformer) or initialize hidden state (for RNN)
        if hasattr(actual_model, 'encode'):
            # Transformer
            # breakpoint()
            memory = actual_model.encode(src, src_len)
        else:
            # RNN: get encoder outputs and hidden
            encoder_outputs, hidden = actual_model.encoder(src, src_len)
        # breakpoint()

        pad_idx = tgt_vocab.pad_idx

        if hasattr(actual_model, 'encode'):
            src_mask = actual_model.make_src_mask(src)
            tgt_mask = actual_model.make_tgt_mask(tgt)
        else:
            src_mask = (src == pad_idx) # [1, src_len]
            if encoder_outputs.size(1) < src_mask.size(1):
                src_mask = src_mask[:, :encoder_outputs.size(1)]

        # decoder_attentions = []

        for _ in range(max_length):
            if hasattr(actual_model, 'decode'):
                # Transformer
                # breakpoint()
                output = actual_model.decode(tgt, memory, src_mask, tgt_mask)
                logits = actual_model.generator(output[:, -1:, :])
            else:
                # RNN
                if len(decoded) == 0:
                    decoder_input = tgt
                else:
                    decoder_input = torch.tensor([[decoded[-1]]], dtype=torch.long, device=device)

                # breakpoint()

                output, hidden, decoder_attention = actual_model.decoder(decoder_input, hidden, encoder_outputs, src_mask)
                logits = actual_model.fc_out(output)

                # breakpoint()

                # decoder_attentions.append(decoder_attention.data)

            # probs = F.softmax(logits, dim=-1)
            # top_probs, top_ids = probs.topk(5)
            # breakpoint()

            # Get most likely token
            token_id = logits.argmax(dim=-1).item()

            # Stop if EOS token
            if token_id == tgt_vocab.eos_idx:
                decoded.append(token_id)
                break

            decoded.append(token_id)

            # Update tgt for transformer
            if hasattr(actual_model, 'decode'):
                tgt = torch.cat([tgt, torch.tensor([[token_id]], dtype=torch.long, device=device)], dim=1)

        # attention_matrix = torch.cat(decoder_attentions)

    # return decoded, attention_matrix
    return decoded


def beam_search_decode(model, src, src_len, tgt_vocab, beam_size: int, max_length: int, device: str, use_ngram_blocking: bool = False) -> List[int]:
    """
    Beam search decoding

    Args:
        model: Translation model
        src: Source sequence (1, src_len)
        src_len: Source length
        tgt_vocab: Target vocabulary
        beam_size: Beam size
        max_length: Maximum decoding length
        device: Device

    Returns:
        Decoded sequence (list of token ids)
    """
    model.eval()

    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        actual_model = model.module
    else:
        actual_model = model

    with torch.no_grad():
        # Encode source
        if hasattr(actual_model, 'encode'):
            # Transformer
            memory = actual_model.encode(src, src_len)  ## for DDP
        else:
            # RNN
            encoder_outputs, hidden = actual_model.encoder(src, src_len)

        # Initialize beam with BOS token
        start_node = BeamSearchNode(
            token_id=tgt_vocab.bos_idx,
            log_prob=0.0,
            hidden_state=hidden if not hasattr(actual_model, 'encode') else None,
            prev_node=None
        )

        beam = [start_node]
        completed = []

        for step in range(max_length):
            candidates = []
            for node in beam:
                # Stop if this is EOS
                if node.token_id == tgt_vocab.eos_idx:
                    completed.append(node)
                    continue

                # Get sequence so far
                sequence = node.get_sequence()
                tgt_input = torch.tensor([sequence], dtype=torch.long, device=device)

                pad_idx = tgt_vocab.pad_idx

                # Get next token probabilities
                if hasattr(actual_model, 'decode'):
                    # Transformer
                    src_mask = actual_model.make_src_mask(src)
                    tgt_mask = actual_model.make_tgt_mask(tgt_input)
                    # breakpoint()
                    output = actual_model.decode(tgt_input, memory, src_mask, tgt_mask)
                    logits = actual_model.generator(output[:, -1:, :])
                else:
                    src_mask = (src == pad_idx) # [1, src_len]
                    # RNN
                    if encoder_outputs.size(1) < src_mask.size(1):
                        src_mask = src_mask[:, :encoder_outputs.size(1)]

                    decoder_input = torch.tensor([[sequence[-1]]], dtype=torch.long, device=device)
                    output, new_hidden, _ = actual_model.decoder(decoder_input, node.hidden_state, encoder_outputs, src_mask)
                    logits = actual_model.fc_out(output)

                log_probs = F.log_softmax(logits.squeeze(0).squeeze(0), dim=-1)

                no_repeat_ngram_size = 3
                if use_ngram_blocking and step >= no_repeat_ngram_size:
                    generated_seq = node.get_sequence()
                    # 只有当序列长度足以构成 n-gram 时才检查
                    if len(generated_seq) >= no_repeat_ngram_size - 1:
                        # 获取用于匹配的后缀（最后 n-1 个词）
                        # 比如我们想避免 "A B C"，现在已有 "A B"，我们要看历史上 "A B" 后面接了什么
                        prefix = generated_seq[-(no_repeat_ngram_size - 1):]
                        
                        # 遍历历史序列，寻找所有出现过 prefix 的位置
                        for i in range(len(generated_seq) - no_repeat_ngram_size + 1):
                            # 检查当前切片是否等于 prefix
                            if generated_seq[i : i + no_repeat_ngram_size - 1] == prefix:
                                # 找到历史中接在 prefix 后面的那个词
                                banned_token = generated_seq[i + no_repeat_ngram_size - 1]
                                # 将该禁止词的概率设为负无穷，强制不被 topk 选中
                                log_probs[banned_token] = -float('inf')

                # Get top-k tokens
                top_log_probs, top_indices = log_probs.topk(beam_size)
                # Create new nodes
                for log_prob, token_id in zip(top_log_probs, top_indices):
                    new_node = BeamSearchNode(
                        token_id=token_id.item(),
                        log_prob=node.log_prob + log_prob.item(),
                        hidden_state=new_hidden if not hasattr(actual_model, 'encode') else None,
                        prev_node=node
                    )
                    candidates.append(new_node)

            # Select top beam_size candidates
            if len(candidates) == 0:
                break

            def gnmt_score(node, alpha=0.7):
                # 5 是经验常数，用来平滑短句子的惩罚
                length = len(node.get_sequence())
                lp = ((5 + length) ** alpha) / ((5 + 1) ** alpha)
                return node.log_prob / lp

            beam = sorted(candidates, key=lambda x: gnmt_score(x), reverse=True)[:beam_size]
            # beam = sorted(candidates, key=lambda x: x.get_avg_log_prob(), reverse=True)[:beam_size]
            # 暂时只看累积概率 (log_prob)，不除以长度
            # beam = sorted(candidates, key=lambda x: x.log_prob, reverse=True)[:beam_size]
            # Stop if all beams have generated EOS
            if len(beam) == 0:
                break

        # Add remaining beams to completed
        completed.extend(beam)
        # Select best sequence
        if len(completed) == 0:
            return []

        best_node = max(completed, key=lambda x: x.get_avg_log_prob())
        sequence = best_node.get_sequence()[1:]  # Remove BOS token
        # Remove EOS token if present
        if len(sequence) > 0 and sequence[-1] == tgt_vocab.eos_idx:
            sequence = sequence[:-1]

        return sequence


if __name__ == "__main__":
    print("Beam search utilities implemented")
    print("Use with model.decode() for Transformer or model.decoder() for RNN")
