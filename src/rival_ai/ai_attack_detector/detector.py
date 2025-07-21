# detector.py

import torch
import torch.nn as nn
import os
import json
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import tempfile
import shutil
import numpy as np
from .models.classifier import AttackClassifier
from .config import Config
from .utils.preprocessing import TextPreprocessor


class AIAttackDetector:
    def __init__(self, model_path=None, config=None, multiclass=False):
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE)
        self.model = None
        self.preprocessor = TextPreprocessor(max_length=self.config.MAX_SEQ_LENGTH)
        self.multiclass = multiclass
        self.num_classes = None
        self.label_mapping = None
        self.contrastive_mode = (
            False  # Track if model was trained with contrastive loss
        )
        self.is_sentence_transformer = (
            True  # Default, will be updated when loading model
        )

        # Only load model if model_path is provided
        if model_path:
            self.load_model(model_path)

    def _is_sentence_transformer_model(self, model_name):
        """
        Determine if a model is a sentence transformer model or a BERT-style model.
        """
        sentence_transformer_indicators = [
            "all-mpnet-base-v2",
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "all-distilroberta-v1",
            "paraphrase-",
            "sentence-transformers/",
        ]

        bert_style_indicators = [
            "answerdotai/ModernBERT",
            "modernbert",
            "bert-",
            "roberta-",
            "deberta-",
        ]

        model_lower = model_name.lower()

        # Check for sentence transformer indicators
        for indicator in sentence_transformer_indicators:
            if indicator.lower() in model_lower:
                return True

        # Check for BERT-style indicators
        for indicator in bert_style_indicators:
            if indicator.lower() in model_lower:
                return False

        # Default assumption - if uncertain, assume sentence transformer for backward compatibility
        return True

    def _create_contrastive_classifier(self, embedding_dim):
        """Create classifier architecture for contrastive models."""
        return nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # No sigmoid - raw logits for contrastive loss
        )

    def load_model(self, model_path):
        """Load a trained model from local path."""
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
        else:
            raise FileNotFoundError("Model config not found")

        # Update instance variables from config
        self.multiclass = model_config.get("multiclass", False)
        self.contrastive_mode = model_config.get("contrastive_mode", False)
        self.is_sentence_transformer = model_config.get("is_sentence_transformer", True)
        base_model_name = model_config.get("base_model", self.config.BASE_MODEL)

        if self.multiclass:
            self.num_classes = model_config.get("num_classes", 26)
            self.label_mapping = {
                int(k): v for k, v in model_config.get("label_mapping", {}).items()
            }

        # Create model with the correct architecture
        self.model = AttackClassifier(
            base_model_name=base_model_name,
            multiclass=self.multiclass,
            num_classes=self.num_classes,
            is_sentence_transformer=self.is_sentence_transformer,
        )

        # If contrastive mode, replace the classifier with the contrastive architecture
        if self.contrastive_mode and not self.multiclass:
            embedding_dim = model_config.get("embedding_dim", 768)
            self.model.classifier = self._create_contrastive_classifier(embedding_dim)

        # Load classification head
        classifier_path = os.path.join(model_path, "classifier.pth")
        self.model.classifier.load_state_dict(
            torch.load(classifier_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        # Update threshold if specified in config (for binary classification)
        if not self.multiclass and "classification_threshold" in model_config:
            self.config.CLASSIFICATION_THRESHOLD = model_config[
                "classification_threshold"
            ]

        print(f"✅ Model loaded successfully")
        print(f"   - Base model: {base_model_name}")
        print(
            f"   - Model type: {'Sentence Transformer' if self.is_sentence_transformer else 'BERT-style'}"
        )
        print(f"   - Multiclass: {self.multiclass}")
        print(f"   - Contrastive mode: {self.contrastive_mode}")
        if self.multiclass:
            print(f"   - Number of classes: {self.num_classes}")

    def setup_model(self):
        """Initialize the model."""
        print(f"Setting up model with base: {self.config.BASE_MODEL}")

        # Determine if this is a sentence transformer or BERT-style model
        self.is_sentence_transformer = self._is_sentence_transformer_model(
            self.config.BASE_MODEL
        )

        self.model = AttackClassifier(
            base_model_name=self.config.BASE_MODEL,
            multiclass=self.multiclass,
            num_classes=self.num_classes,
            is_sentence_transformer=self.is_sentence_transformer,
        )
        self.model.to(self.device)

        print(f"✅ Model setup complete")
        print(f"   - Base model: {self.config.BASE_MODEL}")
        print(
            f"   - Model type: {'Sentence Transformer' if self.is_sentence_transformer else 'BERT-style'}"
        )
        print(f"   - Embedding dim: {self.model.embedding_dim}")
        print(f"   - Device: {self.device}")
        print(f"   - Multi-class: {self.multiclass}")
        if self.multiclass:
            print(f"   - Number of classes: {self.num_classes}")

        return self.model

    @classmethod
    def from_pretrained(cls, multiclass=False, config=None):
        """Load model from HuggingFace Hub based on multiclass parameter."""
        # Create config if not provided
        if config is None:
            config = Config()

        # Determine repo_id based on multiclass parameter
        repo_id = (
            config.HF_MULTICLASS_CLASSIFIER_REPO_ID
            if multiclass
            else config.HF_BINARY_CLASSIFIER_REPO_ID
        )

        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp()

        try:
            print(
                f"Loading {'multiclass' if multiclass else 'binary'} model from: {repo_id}"
            )

            # Download config
            config_path = hf_hub_download(
                repo_id=repo_id, filename="config.json", local_dir=temp_dir
            )

            # Load config
            with open(config_path, "r") as f:
                model_config = json.load(f)

            print(f"Model config loaded: {model_config}")

            # Update config with saved values
            config.BASE_MODEL = model_config["base_model"]
            config.MAX_SEQ_LENGTH = model_config["max_seq_length"]

            # Get model configuration from saved config
            model_multiclass = model_config.get("multiclass", False)
            contrastive_mode = model_config.get("contrastive_mode", False)
            is_sentence_transformer = model_config.get("is_sentence_transformer", True)
            num_classes = model_config.get("num_classes", None)
            label_mapping = {
                int(k): v for k, v in model_config.get("label_mapping", {}).items()
            }

            # Verify that the requested multiclass setting matches the model
            if multiclass != model_multiclass:
                raise ValueError(
                    f"Requested multiclass={multiclass} but model is configured for multiclass={model_multiclass}"
                )

            if not model_multiclass:
                config.CLASSIFICATION_THRESHOLD = model_config[
                    "classification_threshold"
                ]

            # Create detector instance
            detector = cls(config=config, multiclass=model_multiclass)
            detector.num_classes = num_classes
            detector.label_mapping = label_mapping
            detector.contrastive_mode = contrastive_mode
            detector.is_sentence_transformer = is_sentence_transformer

            # Create the model with the correct configuration
            detector.model = AttackClassifier(
                base_model_name=config.BASE_MODEL,
                multiclass=model_multiclass,
                num_classes=num_classes,
                is_sentence_transformer=is_sentence_transformer,
            )

            # If contrastive mode, replace the classifier with the contrastive architecture
            if contrastive_mode and not model_multiclass:
                embedding_dim = model_config.get("embedding_dim", 768)
                detector.model.classifier = detector._create_contrastive_classifier(
                    embedding_dim
                )

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
            print(f"   - Base model: {config.BASE_MODEL}")
            print(
                f"   - Model type: {'Sentence Transformer' if is_sentence_transformer else 'BERT-style'}"
            )
            print(f"   - Multi-class: {model_multiclass}")
            print(f"   - Contrastive mode: {contrastive_mode}")
            if model_multiclass:
                print(f"   - Number of classes: {num_classes}")
                if label_mapping:
                    print(f"   - Label mapping: {label_mapping}")

            return detector

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def predict(self, text):
        """Predict if a text is an attack (binary) or predict class (multiclass)."""
        if self.model is None:
            raise ValueError(
                "No model loaded. Use load_model() or from_pretrained() first."
            )

        # Preprocess text
        processed_text = self.preprocessor.clean_text(text)

        # Get prediction
        with torch.no_grad():
            outputs = self.model([processed_text])

            if self.multiclass:
                # Multi-class prediction
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = int(np.argmax(probs))
                confidence = float(probs[predicted_class])

                result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "class_probabilities": {
                        i: float(prob) for i, prob in enumerate(probs)
                    },
                }

                # Add class name if mapping is available
                if self.label_mapping:
                    result["class_name"] = self.label_mapping.get(predicted_class)
                    result["class_names"] = {
                        i: self.label_mapping[i]
                        for i in range(len(probs))
                        if i in self.label_mapping
                    }

                return result
            else:
                # Binary prediction
                if self.contrastive_mode:
                    # For contrastive models, apply sigmoid to get probabilities
                    prob = torch.sigmoid(outputs.squeeze()).item()
                else:
                    # Standard model already has sigmoid
                    prob = outputs.item()

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
            outputs = self.model(processed_texts)

            if self.multiclass:
                # Multi-class predictions
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                predicted_classes = np.argmax(probs, axis=1)

                results = []
                for i, (pred_class, class_probs) in enumerate(
                    zip(predicted_classes, probs)
                ):
                    confidence = float(class_probs[pred_class])

                    result = {
                        "predicted_class": int(pred_class),
                        "confidence": confidence,
                        "class_probabilities": {
                            j: float(prob) for j, prob in enumerate(class_probs)
                        },
                    }

                    # Add class name if mapping is available
                    if self.label_mapping:
                        result["class_name"] = self.label_mapping.get(pred_class)
                        result["class_names"] = {
                            j: self.label_mapping[j]
                            for j in range(len(class_probs))
                            if j in self.label_mapping
                        }

                    results.append(result)

                return results
            else:
                # Binary predictions
                if self.contrastive_mode:
                    # For contrastive models, apply sigmoid to get probabilities
                    probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                else:
                    # Standard model already has sigmoid
                    probs = outputs.cpu().numpy()

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
        """Set custom classification threshold (only for binary classification)."""
        if self.multiclass:
            print(
                "Warning: Threshold setting is not applicable for multi-class classification."
            )
            return
        self.config.CLASSIFICATION_THRESHOLD = threshold

    def get_label_mapping(self):
        """Get the label mapping dictionary (for multi-class models)."""
        return self.label_mapping

    def get_class_names(self):
        """Get list of class names (for multi-class models)."""
        if not self.multiclass or not self.label_mapping:
            return None
        return [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
