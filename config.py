"""
Configuration file for machine translation project
"""

class Config:
    # Data paths
    # TRAIN_LARGE_PATH = "./dataset/train_100k.jsonl"
    # TRAIN_SMALL_PATH = "./dataset/train_10k.jsonl"
    # VALID_PATH = "./dataset/valid.jsonl"
    # TEST_PATH = "./dataset/test.jsonl"

    TRAIN_LARGE_PATH = "./dataset/train_100k_retranslated_hunyuan.jsonl"
    TRAIN_SMALL_PATH = "./dataset/train_mixed_v2.jsonl"
    VALID_PATH = "./dataset/valid_retranslated_hunyuan.jsonl"
    TEST_PATH = "./dataset/test_retranslated_hunyuan.jsonl"

    # Preprocessing
    MAX_LENGTH = 256 # 50  # Maximum sentence length
    MIN_FREQ = 1 # 2  # Minimum word frequency for vocabulary
    MAX_VOCAB_SIZE = 20000 # 20000 for Trans # 10000 for RNN  # Maximum vocabulary size

    # Special tokens
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'

    # RNN Model
    RNN_EMBED_DIM = 256 # 512
    RNN_HIDDEN_DIM = 512
    RNN_NUM_LAYERS = 2
    RNN_DROPOUT = 0.3
    RNN_CELL_TYPE = 'LSTM'  # 'LSTM' or 'GRU'
    ATTENTION_TYPE = 'additive'  # 'dot', 'multiplicative', 'additive'

    # Transformer Model
    TRANS_D_MODEL = 512 # 256 # 512
    TRANS_NHEAD = 4 # 8
    TRANS_NUM_ENCODER_LAYERS = 4 # 6 # 4 # 4
    TRANS_NUM_DECODER_LAYERS = 4 # 6 # 4 # 4
    TRANS_DIM_FEEDFORWARD = 1024 # 2048
    TRANS_DROPOUT = 0.2 # 0.3 # 0.2 # 0.3
    POSITION_EMBEDDING = 'sinusoidal'  # 'sinusoidal', 'learned', 'relative'
    NORM_TYPE = 'LayerNorm'  # 'LayerNorm', 'RMSNorm'

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    GRAD_CLIP = 5.0
    TEACHER_FORCING_RATIO = 0.5 # 1.0  # 1.0 = always use teacher forcing
    LABEL_SMOOTHING = 0.1
    WARMUP_STEPS = 4000

    # Decoding
    BEAM_SIZE = 5
    MAX_DECODE_LENGTH = 60

    # Device
    DEVICE = 'cuda'  # 'cuda' or 'cpu'

    # Checkpoints
    RNN_CHECKPOINT_DIR = "./checkpoints/rnn"
    TRANS_CHECKPOINT_DIR = "./checkpoints/transformer"

    # Evaluation
    EVAL_EVERY = 1000  # Evaluate every N batches
    SAVE_EVERY = 5000  # Save checkpoint every N batches

    # Teaching forcing decay
    TEACHER_FORCING_RATIO = 1.0  # Decay teacher forcing ratio every epoch
    DECAY_RATE = 0.05  # Decay rate for teacher forcing ratio
    MIN_RATIO = 0.0  # Minimum teacher forcing ratio
