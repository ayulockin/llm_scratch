# Dataset
ORIGINAL_DATASET = "wmt/wmt14"
DE_EN_SPLIT_DATASET = "llm-scratch/wmt14-de-en-split"

# Tokenizer
TOKENIZER_ID = "meta-llama/Llama-2-7b"

# Data
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16

# Model
MODEL_DIM = 512
EXPANSION_DIM = 2048
NUM_HEADS = 8
NUM_BLOCKS = 6
DROPOUT_RATE = 0.1
MODEL_NAME = "llm-scratch/wmt-14-de-en-model"

# Train
NUM_EPOCHS = 2
NUM_TOKENS = 2e8  # 200M tokens
TOTAL_TOKENS_IN_DATASET = 1.94e8  # 194M tokens - source is `en`. Run with `CALCULATE_TOTAL_TOKENS=True` to verify.
VALIDATION_STEP = 10000
WARMUP_PERCENTAGE = 0.4
LEARNING_RATE = 1.0
BETAS = (0.9, 0.98)
EPSILON = 1e-09
GRAD_ACCUMULATION_STEP = 1
WANDD_ENTITY = "llm-scratch"
WANDB_PROJECT = "wmt14-de-en"

# Debug
CALCULATE_TOTAL_TOKENS = False
