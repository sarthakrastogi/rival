import torch
import os
import json
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import tempfile
import shutil
from .models.classifier import AttackClassifier
from .config import Config
from .utils.preprocessing import TextPreprocessor


class AIAttackDetector:
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE)
        self.model = None
        self.preprocessor = TextPreprocessor(max_length=self.config.MAX_SEQ_LENGTH)

        # Only load model if model_path is provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load a trained model from local path."""
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
        else:
            raise FileNotFoundError("Model config not found")

        # Load sentence transformer
        st_path = os.path.join(model_path, "sentence_transformer")
        sentence_transformer = SentenceTransformer(st_path)

        # Create model
        self.model = AttackClassifier()
        self.model.sentence_transformer = sentence_transformer

        # Load classification head
        classifier_path = os.path.join(model_path, "classifier.pth")
        self.model.classifier.load_state_dict(
            torch.load(classifier_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        # Update threshold if specified in config
        if "classification_threshold" in model_config:
            self.config.CLASSIFICATION_THRESHOLD = model_config[
                "classification_threshold"
            ]

    def setup_model(self):
        """Initialize the model."""
        print(f"Setting up model with base: {self.config.BASE_MODEL}")

        self.model = AttackClassifier(self.config.BASE_MODEL)
        self.model.to(self.device)

        # Freeze sentence transformer parameters (only during training)
        # During inference, we don't need to freeze anything

        print(f"✅ Model setup complete")
        print(f"   - Base model: {self.config.BASE_MODEL}")
        print(f"   - Embedding dim: {self.model.embedding_dim}")
        print(f"   - Device: {self.device}")

        return self.model

    @classmethod
    def from_pretrained(
        cls, repo_id="sarthakrastogi/rival_ai_attack_detector", config=None
    ):
        """Load model from HuggingFace Hub or local path."""
        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp()

        try:
            print(f"Loading model from: {repo_id}")

            # Download config
            config_path = hf_hub_download(
                repo_id=repo_id, filename="config.json", local_dir=temp_dir
            )

            # Load config
            with open(config_path, "r") as f:
                model_config = json.load(f)

            print(f"Model config loaded: {model_config}")

            # Create config object
            if config is None:
                config = Config()

            # Update config with saved values
            config.BASE_MODEL = model_config["base_model"]
            config.CLASSIFICATION_THRESHOLD = model_config["classification_threshold"]
            config.MAX_SEQ_LENGTH = model_config["max_seq_length"]

            # Create detector instance
            detector = cls(config=config)  # Don't pass model_path here

            # Create the model with the correct base model
            detector.model = AttackClassifier(config.BASE_MODEL)
            detector.model.to(detector.device)

            # Download and load classifier weights
            classifier_path = hf_hub_download(
                repo_id=repo_id, filename="classifier.pth", local_dir=temp_dir
            )

            classifier_state = torch.load(classifier_path, map_location=detector.device)
            detector.model.classifier.load_state_dict(classifier_state)

            # Set model to eval mode
            detector.model.eval()

            print(f"✅ Model loaded successfully from {repo_id}")
            return detector

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def predict(self, text):
        """Predict if a text is an attack."""
        if self.model is None:
            raise ValueError(
                "No model loaded. Use load_model() or from_pretrained() first."
            )

        # Preprocess text
        processed_text = self.preprocessor.clean_text(text)

        # Get prediction
        with torch.no_grad():
            prob = self.model([processed_text]).item()

        is_attack = prob > self.config.CLASSIFICATION_THRESHOLD

        return {
            "is_attack": is_attack,
            "confidence": prob,
            "threshold": self.config.CLASSIFICATION_THRESHOLD,
        }

    def predict_batch(self, texts):
        """Predict for a batch of texts."""
        if self.model is None:
            raise ValueError(
                "No model loaded. Use load_model() or from_pretrained() first."
            )

        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_batch(texts)

        # Get predictions
        with torch.no_grad():
            probs = self.model(processed_texts).cpu().numpy()

        results = []
        for prob in probs:
            is_attack = prob > self.config.CLASSIFICATION_THRESHOLD
            results.append(
                {
                    "is_attack": is_attack,
                    "confidence": float(prob),
                    "threshold": self.config.CLASSIFICATION_THRESHOLD,
                }
            )

        return results

    def set_threshold(self, threshold):
        """Set custom classification threshold."""
        self.config.CLASSIFICATION_THRESHOLD = threshold
