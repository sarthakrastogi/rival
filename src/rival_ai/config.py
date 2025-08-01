"""Configuration settings for the Rival AI."""

import os
import torch


class Config:
    """Configuration class for AI settings."""

    def __init__(self):
        # General API configuration
        self.api_base_url = os.getenv("RIVAL_API_BASE_URL", "http://localhost:8000")
        self.default_model = os.getenv("RIVAL_DEFAULT_MODEL", "gpt-4.1")
        self.default_limit = int(os.getenv("RIVAL_DEFAULT_LIMIT", "100"))

        # AI Attack Detector Model configuration
        self.BASE_MODEL = "answerdotai/ModernBERT-large"  # "all-mpnet-base-v2"  # Powerful sentence transformer model
        self.MAX_SEQ_LENGTH = 512
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS = 5
        self.LEARNING_RATE = 2e-5

        # Training configuration
        self.TRAIN_TEST_SPLIT = 0.8
        self.RANDOM_SEED = 42

        # Device configuration
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # HuggingFace configuration
        self.HF_BINARY_CLASSIFIER_MODEL_NAME = "rival_ai_attack_detector_binary"
        self.HF_MULTICLASS_CLASSIFIER_MODEL_NAME = "rival_ai_attack_detector_multiclass"
        self.HF_ORGANIZATION = "sarthakrastogi"
        self.HF_BINARY_CLASSIFIER_REPO_ID = (
            f"{self.HF_ORGANIZATION}/{self.HF_BINARY_CLASSIFIER_MODEL_NAME}"
        )
        self.HF_MULTICLASS_CLASSIFIER_REPO_ID = (
            f"{self.HF_ORGANIZATION}/{self.HF_MULTICLASS_CLASSIFIER_MODEL_NAME}"
        )

        # Classification thresholds
        self.CLASSIFICATION_THRESHOLD = (
            0.519  # Default threshold for binary classification
        )

        # Data configuration
        self.TEXT_COLUMN = "text"
        self.LABEL_COLUMN = "label"  # 0 for benign, 1 for attack

    @property
    def api_endpoints(self) -> dict:
        """Get API endpoints configuration."""
        return {
            "testcases": f"{self.api_base_url}/sdk/v1/agent/testcases",
            "evaluate": f"{self.api_base_url}/sdk/v1/evaluate/testcase",
            "save_agent": f"{self.api_base_url}/sdk/v1/agents/save",
        }


# Global config instance
config = Config()
