"""
Evaluation metrics for machine translation
Implements BLEU-4 and precision_n metrics
"""

import math
from collections import Counter
from typing import List
# import sacrebleu
import nltk
from nltk.translate.bleu_score import modified_precision, SmoothingFunction
# from nltk.translate.bleu_score import corpus_bleu


chencherry = SmoothingFunction()


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return Counter(ngrams)


# def precision_n(reference: List[str], hypothesis: List[str], n: int) -> float:
#     """
#     Calculate n-gram precision

#     Args:
#         reference: Reference tokens
#         hypothesis: Hypothesis tokens
#         n: n-gram order

#     Returns:
#         Precision score
#     """
#     if len(hypothesis) < n:
#         return 0.0

#     ref_ngrams = get_ngrams(reference, n)
#     hyp_ngrams = get_ngrams(hypothesis, n)

#     # Count matches (clipped)
#     matches = 0
#     for ngram, count in hyp_ngrams.items():
#         matches += min(count, ref_ngrams.get(ngram, 0))

#     # Total hypothesis n-grams
#     total = sum(hyp_ngrams.values())

#     return matches / total if total > 0 else 0.0


def precision_n(reference: List[str], hypothesis: List[str], n: int) -> float:
    """
    使用 NLTK 计算单句的 Modified Precision (n-gram 精度)。
    替代了你原本的手动 n-gram 统计和 clipping 逻辑。
    """
    
    # 1. 格式适配
    # NLTK 的 modified_precision 函数要求 reference 是一个列表的列表 (List[List[str]])
    # 因为标准 BLEU 支持一个句子有多个参考答案。
    # 即便你只有一个参考，也必须包裹一层：['a', 'b'] -> [['a', 'b']]
    list_of_references = [reference]
    
    # 2. 调用 NLTK
    # 返回值是一个 fractions.Fraction 对象 (例如 3/4)，这非常精确
    fraction = modified_precision(list_of_references, hypothesis, n)

    # breakpoint()
    
    # 3. 转换为浮点数并处理分母为 0 的情况
    # 如果 hypothesis 长度小于 n，或者没有 n-gram，分母为 0
    if fraction.denominator == 0:
        return 0.0
    
    # return float(fraction)
    return fraction.numerator, fraction.denominator


def brevity_penalty(reference: List[str], hypothesis: List[str]) -> float:
    """
    Calculate brevity penalty for BLEU

    Args:
        reference: Reference tokens
        hypothesis: Hypothesis tokens

    Returns:
        Brevity penalty
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if hyp_len >= ref_len:
        return 1.0
    else:
        return math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0


# def bleu_score(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
#     """
#     Calculate BLEU score

#     Args:
#         reference: Reference tokens
#         hypothesis: Hypothesis tokens
#         max_n: Maximum n-gram order (default: 4 for BLEU-4)

#     Returns:
#         BLEU score
#     """
#     # Calculate precision for each n-gram order
#     precisions = []
#     for n in range(1, max_n + 1):
#         p = precision_n(reference, hypothesis, n)
#         if p == 0:
#             # Use smoothing to avoid log(0)
#             p = 1e-10
#         precisions.append(p)

#     # Geometric mean of precisions
#     geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

#     # Apply brevity penalty
#     bp = brevity_penalty(reference, hypothesis)

#     return bp * geo_mean


def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    """
    Calculate corpus-level BLEU score

    Args:
        references: List of reference token lists
        hypotheses: List of hypothesis token lists
        max_n: Maximum n-gram order

    Returns:
        Corpus BLEU score
    """
    # NLTK expects references to be a list of lists of lists 
    # (because each hypothesis can have multiple valid reference translations).
    # Since you likely have 1 reference per sentence, we wrap each ref in a list.
    # Input:  [['the', 'cat'], ['hello']]
    # Output: [[['the', 'cat']], [['hello']]]
    list_of_references = [[ref] for ref in references]
    
    # Calculate BLEU-4
    # weights=(0.25, 0.25, 0.25, 0.25) means equal weight for 1, 2, 3, and 4-grams.
    score = nltk.translate.bleu_score.corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    
    # breakpoint()

    return score


# def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
#     """
#     Calculate corpus-level BLEU score

#     Args:
#         references: List of reference token lists
#         hypotheses: List of hypothesis token lists
#         max_n: Maximum n-gram order

#     Returns:
#         Corpus BLEU score
#     """
#     # Accumulate n-gram statistics
#     total_matches = [0] * max_n
#     total_possible = [0] * max_n
#     total_ref_len = 0
#     total_hyp_len = 0

#     for ref, hyp in zip(references, hypotheses):
#         total_ref_len += len(ref)
#         total_hyp_len += len(hyp)

#         for n in range(1, max_n + 1):
#             ref_ngrams = get_ngrams(ref, n)
#             hyp_ngrams = get_ngrams(hyp, n)

#             # Count matches
#             matches = 0
#             for ngram, count in hyp_ngrams.items():
#                 matches += min(count, ref_ngrams.get(ngram, 0))

#             total_matches[n-1] += matches
#             total_possible[n-1] += max(len(hyp) - n + 1, 0)

#     # Calculate precisions
#     precisions = []
#     for matches, possible in zip(total_matches, total_possible):
#         if possible == 0:
#             p = 0.0
#         else:
#             p = matches / possible
#             if p == 0:
#                 p = 1e-10  # Smoothing
#         precisions.append(p)

#     # Geometric mean
#     geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

#     # Brevity penalty
#     if total_hyp_len >= total_ref_len:
#         bp = 1.0
#     else:
#         bp = math.exp(1 - total_ref_len / total_hyp_len) if total_hyp_len > 0 else 0.0

#     return bp * geo_mean


# def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
#     """
#     Calculate corpus-level BLEU score

#     Args:
#         references: List of reference token lists
#         hypotheses: List of hypothesis token lists
#         max_n: Maximum n-gram order

#     Returns:
#         Corpus BLEU score
#     """
#     hypotheses = [' '.join(hyp) for hyp in hypotheses]
#     references = [[' '.join(ref)] for ref in references]
#     print(hypotheses[0])
#     print(references[0])
#     bleu = sacrebleu.corpus_bleu(hypotheses, references)
#     return bleu.score


# import math
# from collections import Counter
# from typing import List

# def corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
#     """
#     Calculates BLEU-4 based on the custom definition:
#     Score = min(1, output_len / ref_len) * (p1 * p2 * p3 * p4)
#     """
#     # 1. Initialize Counters
#     total_matches = [0] * max_n
#     total_possible = [0] * max_n
#     total_ref_len = 0
#     total_hyp_len = 0

#     # 2. Collect Statistics (Same as your original code)
#     for ref, hyp in zip(references, hypotheses):
#         # breakpoint()
#         total_ref_len += len(ref)
#         total_hyp_len += len(hyp)

#         for n in range(1, max_n + 1):
#             # Get n-grams
#             ref_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref)-n+1)])
#             hyp_ngrams = Counter([tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1)])

#             # Count matches (clipped by reference count)
#             matches = 0
#             for ngram, count in hyp_ngrams.items():
#                 matches += min(count, ref_ngrams.get(ngram, 0))
            
#             total_matches[n-1] += matches
#             total_possible[n-1] += max(len(hyp) - n + 1, 0)

#     # 3. Calculate Precisions (p1, p2, p3, p4)
#     precisions = []
#     for matches, possible in zip(total_matches, total_possible):
#         if possible == 0:
#             precisions.append(0.0)
#         else:
#             precisions.append(matches / possible)

#     # --- CHANGED: Apply Your Custom Formula ---

#     # Term 1: The Product (Pi symbol in your image)
#     # Formula: p1 * p2 * p3 * p4
#     precision_product = 1.0
#     for p in precisions:
#         precision_product *= p

#     # Term 2: The Brevity Penalty (Min term in your image)
#     # Formula: min(1, output-length / reference-length)
#     if total_ref_len == 0:
#         bp = 1.0
#     else:
#         bp = min(1.0, total_hyp_len / total_ref_len)

#     # Final Score
#     return bp * precision_product


def calculate_all_metrics(references: List[List[str]], hypotheses: List[List[str]]) -> dict:
    """
    Calculate all metrics: BLEU-4 and precision_1 to precision_4

    Args:
        references: List of reference token lists
        hypotheses: List of hypothesis token lists

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # BLEU-4
    metrics['bleu4'] = corpus_bleu(references, hypotheses, max_n=4)

    # Precision_n for n=1,2,3,4
    # for n in range(1, 5):
    #     total_precision = 0.0
    #     count = 0
    #     for ref, hyp in zip(references, hypotheses):
    #         p = precision_n(ref, hyp, n)
    #         total_precision += p
    #         count += 1
    #     metrics[f'precision_{n}'] = total_precision / count if count > 0 else 0.0
        
    # metrics[f'precision_{n}'] = score

    for n in range(1, 5):
        total_numerator = 0
        total_denominator = 0
        for ref, hyp in zip(references, hypotheses):
            numerator, denominator = precision_n(ref, hyp, n)
            total_numerator += numerator
            total_denominator += denominator
        if total_denominator == 0:
            metrics[f'precision_{n}'] = 0.0
        else:
            metrics[f'precision_{n}'] = total_numerator / total_denominator

    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics\n")

    # Example 1: Perfect match
    ref1 = ["the", "cat", "sat", "on", "the", "mat"]
    hyp1 = ["the", "cat", "sat", "on", "the", "mat"]
    print("Example 1: Perfect match")
    print(f"Reference: {' '.join(ref1)}")
    print(f"Hypothesis: {' '.join(hyp1)}")
    print(f"BLEU-4: {bleu_score(ref1, hyp1):.4f}")
    for n in range(1, 5):
        print(f"Precision-{n}: {precision_n(ref1, hyp1, n):.4f}")

    # Example 2: Partial match
    ref2 = ["the", "cat", "sat", "on", "the", "mat"]
    hyp2 = ["the", "dog", "sat", "on", "a", "mat"]
    print("\nExample 2: Partial match")
    print(f"Reference: {' '.join(ref2)}")
    print(f"Hypothesis: {' '.join(hyp2)}")
    print(f"BLEU-4: {bleu_score(ref2, hyp2):.4f}")
    for n in range(1, 5):
        print(f"Precision-{n}: {precision_n(ref2, hyp2, n):.4f}")

    # Example 3: Corpus-level
    refs = [ref1, ref2]
    hyps = [hyp1, hyp2]
    print("\nExample 3: Corpus-level metrics")
    metrics = calculate_all_metrics(refs, hyps)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # broken metric test
    # refs = [["the", "cat", "is", "on", "the", "mat"]]
    refs = ' '.join(["the", "cat", "is", "on", "the", "mat"])

    # # Dummy Hypothesis: "The cat is on the" (Missing word -> Penalty)
    # # hyps_bad = [["the", "cat", "is", "on", "the"]]
    # hyps_bad = ' '.join(["the", "cat", "is", "on", "the"])
    # # print(refs, hyps_bad)
    # # print(hyps_bad == "the cat is on the")

    # score = corpus_bleu([[refs]], [hyps_bad])
    # print(f"Test Score: {score}")
