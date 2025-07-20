import torch


class Config:
    # Model configuration
    BASE_MODEL = "all-mpnet-base-v2"  # Powerful sentence transformer model
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5

    # Training configuration
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_SEED = 42

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # HuggingFace configuration
    HF_MODEL_NAME = "rival_ai_attack_detector"
    HF_ORGANIZATION = "sarthakrastogi"
    HF_REPO_ID = f"{HF_ORGANIZATION}/{HF_MODEL_NAME}"

    # Classification thresholds
    CLASSIFICATION_THRESHOLD = 0.5

    # Data configuration
    TEXT_COLUMN = "text"
    LABEL_COLUMN = "label"  # 0 for benign, 1 for attack
