# Dataset
ORIGINAL_DATASET = "wmt/wmt14"
DE_EN_SPLIT_DATASET = "llm-scratch/wmt14-de-en-split"

# Tokenizer
VOCAB_SIZE = 37_000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "begin_token": "[BOS]",
    "end_token": "[EOS]",
    "unknown_token": "[UNK]",
}
TOKENIZER_ID = "llm-scratch/wmt-14-de-en-tokenizer"

# Data
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 8

# Model
MODEL_DIM = 512
EXPANSION_DIM = 2048
NUM_HEADS = 8
NUM_BLOCKS = 6
DROPOUT_RATE = 0.1
MODEL_NAME = "llm-scratch/wmt-14-de-en-model"

# Train
NUM_EPOCHS = 2
WARMUP_PERCENTAGE = 0.4
LEARNING_RATE = 1.0
BETAS = (0.9, 0.98)
EPSILON = 1e-09
GRAD_ACCUMULATION_STEP = 1
WANDD_ENTITY = "llm-scratch"
WANDB_PROJECT = "wmt14-de-en"