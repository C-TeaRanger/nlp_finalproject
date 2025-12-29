"""
One-click inference script for Chinese-English Machine Translation
Supports RNN and Transformer models with greedy and beam search decoding

Usage:
    # Interactive mode
    python inference.py --model rnn --checkpoint ./checkpoints/rnn/best_model.pt
    
    # Translate a single sentence
    python inference.py --model rnn --checkpoint ./checkpoints/rnn/best_model.pt --text "这是一个测试句子"
    
    # Translate from file
    python inference.py --model transformer --checkpoint ./checkpoints/transformer/best_model.pt --input input.txt --output output.txt
    
    # Use beam search
    python inference.py --model rnn --checkpoint ./checkpoints/rnn/best_model.pt --text "你好世界" --beam-size 5
"""

import torch
import os
import sys
import argparse
import json
from collections import OrderedDict
from typing import List, Optional, Union

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.rnn.seq2seq import Seq2Seq
from models.transformer.transformer import Transformer
from utils.beam_search import greedy_decode, beam_search_decode
from data.vocabulary_bpe_en import BPEVocabularyEN
from data.vocabulary_bpe_zh import BPEVocabularyZH


class Translator:
    """Unified translator class for both RNN and Transformer models"""
    
    def __init__(self, 
                 model_type: str,
                 checkpoint_path: str,
                 device: str = None,
                 config: Config = None):
        """
        Initialize the translator
        
        Args:
            model_type: 'rnn' or 'transformer'
            checkpoint_path: Path to model checkpoint
            device: 'cuda' or 'cpu' (auto-detected if None)
            config: Config object (created if None)
        """
        self.model_type = model_type.lower()
        self.config = config or Config()
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model and vocabularies
        if self.model_type == 'rnn':
            self.model, self.vocab_zh, self.vocab_en = self._load_rnn_model(checkpoint_path)
        elif self.model_type == 'transformer':
            self.model, self.vocab_zh, self.vocab_en = self._load_transformer_model(checkpoint_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'rnn' or 'transformer'.")
        
        self.model.eval()
        print(f"Model loaded successfully!")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_rnn_model(self, checkpoint_path: str):
        """Load RNN model from checkpoint"""
        # Determine checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not checkpoint_dir:
            checkpoint_dir = "./checkpoints/rnn/"
        
        # Load vocabularies
        vocab_zh = BPEVocabularyZH("Chinese")
        vocab_en = BPEVocabularyEN("English")
        
        tokenizer_zh_path = os.path.join(checkpoint_dir, 'tokenizer_zh.pkl')
        tokenizer_en_path = os.path.join(checkpoint_dir, 'tokenizer_en.pkl')
        
        if os.path.exists(tokenizer_zh_path):
            vocab_zh.load(tokenizer_zh_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_zh_path}")
        
        if os.path.exists(tokenizer_en_path):
            vocab_en.load(tokenizer_en_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_en_path}")
        
        print("Vocabularies loaded.")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DDP wrapped state dict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith("module."):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        
        # Create model
        model = Seq2Seq(
            src_vocab_size=len(vocab_zh),
            tgt_vocab_size=len(vocab_en),
            embed_dim=self.config.RNN_EMBED_DIM,
            hidden_dim=self.config.RNN_HIDDEN_DIM,
            num_layers=self.config.RNN_NUM_LAYERS,
            dropout=self.config.RNN_DROPOUT,
            cell_type=self.config.RNN_CELL_TYPE,
            attention_type=self.config.ATTENTION_TYPE,
            pad_idx=vocab_en.pad_idx
        ).to(self.device)
        
        model.load_state_dict(new_state_dict)
        
        return model, vocab_zh, vocab_en
    
    def _load_transformer_model(self, checkpoint_path: str):
        """Load Transformer model from checkpoint"""
        # Determine checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not checkpoint_dir:
            checkpoint_dir = "./checkpoints/transformer/"
        
        # Load vocabularies
        vocab_zh = BPEVocabularyZH("Chinese")
        vocab_en = BPEVocabularyEN("English")
        
        tokenizer_zh_path = os.path.join(checkpoint_dir, 'tokenizer_zh.pkl')
        tokenizer_en_path = os.path.join(checkpoint_dir, 'tokenizer_en.pkl')
        
        if os.path.exists(tokenizer_zh_path):
            vocab_zh.load(tokenizer_zh_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_zh_path}")
        
        if os.path.exists(tokenizer_en_path):
            vocab_en.load(tokenizer_en_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_en_path}")
        
        print("Vocabularies loaded.")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DDP wrapped state dict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith("module."):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        
        # Create model
        model = Transformer(
            src_vocab_size=len(vocab_zh),
            tgt_vocab_size=len(vocab_en),
            d_model=self.config.TRANS_D_MODEL,
            num_heads=self.config.TRANS_NHEAD,
            num_encoder_layers=self.config.TRANS_NUM_ENCODER_LAYERS,
            num_decoder_layers=self.config.TRANS_NUM_DECODER_LAYERS,
            dim_feedforward=self.config.TRANS_DIM_FEEDFORWARD,
            dropout=self.config.TRANS_DROPOUT,
            position_embedding=self.config.POSITION_EMBEDDING,
            norm_type=self.config.NORM_TYPE,
            pad_idx=vocab_en.pad_idx
        ).to(self.device)
        
        model.load_state_dict(new_state_dict)
        
        return model, vocab_zh, vocab_en
    
    def translate(self, 
                  text: str,
                  beam_size: int = 1,
                  max_length: int = None) -> str:
        """
        Translate a single Chinese sentence to English
        
        Args:
            text: Chinese text to translate
            beam_size: Beam size for decoding (1 = greedy)
            max_length: Maximum output length
            
        Returns:
            English translation
        """
        if max_length is None:
            max_length = self.config.MAX_DECODE_LENGTH
        
        # Tokenize input
        src_ids = self.vocab_zh.encode(text)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if beam_size <= 1:
                # Greedy decoding
                pred_ids = greedy_decode(
                    self.model, src_tensor, src_len, self.vocab_en,
                    max_length, str(self.device)
                )
            else:
                # Beam search decoding
                pred_ids = beam_search_decode(
                    self.model, src_tensor, src_len, self.vocab_en,
                    beam_size, max_length, str(self.device)
                )
        
        # Decode to string
        translation = self.vocab_en.decode(pred_ids, skip_special=True)
        
        return translation
    
    def translate_batch(self, 
                        texts: List[str],
                        beam_size: int = 1,
                        max_length: int = None) -> List[str]:
        """
        Translate a batch of Chinese sentences to English
        
        Args:
            texts: List of Chinese texts to translate
            beam_size: Beam size for decoding
            max_length: Maximum output length
            
        Returns:
            List of English translations
        """
        translations = []
        for text in texts:
            translation = self.translate(text, beam_size, max_length)
            translations.append(translation)
        return translations
    
    def translate_file(self,
                       input_path: str,
                       output_path: str,
                       beam_size: int = 1,
                       max_length: int = None):
        """
        Translate all sentences in a file
        
        Args:
            input_path: Path to input file (one sentence per line)
            output_path: Path to output file
            beam_size: Beam size for decoding
            max_length: Maximum output length
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Translating {len(lines)} sentences...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines):
                translation = self.translate(line, beam_size, max_length)
                f.write(translation + '\n')
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(lines)} sentences")
        
        print(f"Translations saved to {output_path}")


def interactive_mode(translator: Translator, beam_size: int = 1):
    """Run interactive translation mode"""
    print("\n" + "=" * 60)
    print("Interactive Translation Mode")
    print("Enter Chinese text to translate. Type 'quit' to exit.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            text = input("Chinese: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        translation = translator.translate(text, beam_size=beam_size)
        print(f"English: {translation}\n")


def main():
    parser = argparse.ArgumentParser(
        description="One-click inference for Chinese-English Machine Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with RNN model
  python inference.py --model rnn --checkpoint ./checkpoints/rnn/best_model.pt
  
  # Translate single sentence
  python inference.py --model transformer --checkpoint ./checkpoints/transformer/best_model.pt --text "你好世界"
  
  # Translate file with beam search
  python inference.py --model rnn --checkpoint ./checkpoints/rnn/best_model.pt --input input.txt --output output.txt --beam-size 5
        """
    )
    
    parser.add_argument('--model', type=str, required=True, choices=['rnn', 'transformer'],
                        help='Model type: rnn or transformer')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file')
    parser.add_argument('--text', type=str,
                        help='Chinese text to translate (single sentence mode)')
    parser.add_argument('--input', type=str,
                        help='Input file path (file mode, one sentence per line)')
    parser.add_argument('--output', type=str,
                        help='Output file path (required if --input is specified)')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size for decoding (default: 1 = greedy)')
    parser.add_argument('--max-length', type=int, default=60,
                        help='Maximum output length (default: 60)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Initialize translator
    print("Loading model...")
    translator = Translator(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Run appropriate mode
    if args.text:
        # Single sentence mode
        translation = translator.translate(args.text, beam_size=args.beam_size, max_length=args.max_length)
        print(f"\nChinese: {args.text}")
        print(f"English: {translation}")
        
    elif args.input:
        # File mode
        translator.translate_file(args.input, args.output, beam_size=args.beam_size, max_length=args.max_length)
        
    else:
        # Interactive mode
        interactive_mode(translator, beam_size=args.beam_size)


if __name__ == "__main__":
    main()
