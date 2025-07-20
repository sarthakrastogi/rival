import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import numpy as np
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo
import os
import json

from .classifier import AttackClassifier
from ..config import Config
from ..utils.preprocessing import TextPreprocessor

# Check if token is in environment
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
print(f"Token found: {token is not None}")

if token:
    # Test the token
    try:
        api = HfApi(token=token)
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception as e:
        print(f"Token authentication failed: {e}")


class AttackDetectorTrainer:
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE)
        self.model = None
        self.preprocessor = TextPreprocessor(max_length=self.config.MAX_SEQ_LENGTH)

    def setup_model(self):
        """Initialize the model."""
        self.model = AttackClassifier(self.config.BASE_MODEL)
        self.model.to(self.device)

        # Only train the classification head, keep sentence transformer frozen
        for param in self.model.sentence_transformer.parameters():
            param.requires_grad = False

        return self.model

    def _preprocess_dataset(self, dataset):
        """Convert text dataset to embeddings dataset for faster training."""
        print("Preprocessing dataset - converting texts to embeddings...")

        if self.model is None:
            self.setup_model()

        self.model.eval()  # Set to eval mode for preprocessing

        all_embeddings = []
        all_labels = []

        # Process in batches to avoid memory issues
        batch_size = 32
        for i in tqdm(range(0, len(dataset), batch_size)):
            # Get actual batch end index
            batch_end = min(i + batch_size, len(dataset))

            # Collect texts and labels for this batch
            batch_texts = []
            batch_labels = []

            for j in range(i, batch_end):
                sample = dataset[j]  # This returns {"text": str, "label": float}
                batch_texts.append(sample["text"])
                batch_labels.append(sample["label"])

            # Preprocess texts
            texts = self.preprocessor.preprocess_batch(batch_texts)

            # Get embeddings (no gradients needed for preprocessing)
            with torch.no_grad():
                embeddings = self.model.sentence_transformer.encode(
                    texts, convert_to_tensor=True, device=self.device
                )
                all_embeddings.append(embeddings.cpu())

                # Convert labels to tensor
                labels_tensor = torch.tensor(batch_labels, dtype=torch.float32)
                all_labels.append(labels_tensor)

        # Concatenate all embeddings and labels
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        return TensorDataset(embeddings_tensor, labels_tensor)

    def train(self, train_dataset, test_dataset):
        """Train the model with preprocessed embeddings."""
        if self.model is None:
            self.setup_model()

        # Convert datasets to embedding datasets
        train_embedding_dataset = self._preprocess_dataset(train_dataset)
        test_embedding_dataset = self._preprocess_dataset(test_dataset)

        # Create data loaders
        train_loader = TorchDataLoader(
            train_embedding_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True
        )
        test_loader = TorchDataLoader(
            test_embedding_dataset, batch_size=self.config.BATCH_SIZE
        )

        # Setup optimizer and loss (only for classifier head)
        optimizer = optim.Adam(
            self.model.classifier.parameters(),  # Only classifier parameters
            lr=self.config.LEARNING_RATE,
        )
        criterion = nn.BCELoss()

        # Training loop
        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            total_loss = 0

            for embeddings, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}"
            ):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass through classifier only
                outputs = self.model.classifier(embeddings).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Evaluate on test set
            test_metrics = self.evaluate_embeddings(test_loader)

            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")
            print("-" * 50)

    def evaluate_embeddings(self, test_loader):
        """Evaluate the model on preprocessed embeddings."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings = embeddings.to(self.device)

                outputs = self.model.classifier(embeddings).squeeze()
                probs = outputs.cpu().numpy()
                preds = (probs > self.config.CLASSIFICATION_THRESHOLD).astype(int)

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary"
        )
        auc = roc_auc_score(all_labels, all_probs)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

    def evaluate(self, test_loader):
        """Evaluate the model on text data (for compatibility)."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                texts = batch["text"]
                labels = batch["label"]

                # Preprocess texts
                texts = self.preprocessor.preprocess_batch(texts)

                outputs = self.model(texts)
                probs = outputs.cpu().numpy()
                preds = (probs > self.config.CLASSIFICATION_THRESHOLD).astype(int)

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary"
        )
        auc = roc_auc_score(all_labels, all_probs)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

    def save_model(self, save_path):
        """Save the trained model with proper sentence transformer handling."""
        os.makedirs(save_path, exist_ok=True)

        # Save the classification head
        torch.save(
            self.model.classifier.state_dict(),
            os.path.join(save_path, "classifier.pth"),
        )

        # Save the sentence transformer model name instead of the model itself
        # This way we can recreate it during loading
        config_dict = {
            "base_model": self.config.BASE_MODEL,
            "embedding_dim": self.model.embedding_dim,
            "classification_threshold": self.config.CLASSIFICATION_THRESHOLD,
            "max_seq_length": self.config.MAX_SEQ_LENGTH,
        }

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Model saved to {save_path}")
        print(f"   - classifier.pth: Classification head weights")
        print(f"   - config.json: Model configuration")

    def push_to_hub(self, repo_name, organization=None, private=False):
        """Push model to HuggingFace Hub with improved error handling."""
        import os
        import shutil
        from huggingface_hub import HfApi, create_repo

        # Get token from environment
        token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "No HuggingFace token found. Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN environment variable."
            )

        # Create temporary directory for model files
        temp_dir = "temp_model"
        self.save_model(temp_dir)

        # Create repository
        if organization:
            full_repo_name = f"{organization}/{repo_name}"
        else:
            full_repo_name = repo_name

        # Initialize API with token
        api = HfApi(token=token)

        try:
            # Try to create repository (will fail if it already exists, which is fine)
            create_repo(full_repo_name, private=private, exist_ok=True, token=token)
            print(f"✅ Repository {full_repo_name} ready")
        except Exception as e:
            print(f"Repository creation info: {e}")
            # Continue anyway - repository might already exist

        try:
            # Upload all files in the temp directory
            for file_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file_name)
                if os.path.isfile(file_path):
                    print(f"Uploading {file_name}...")
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=file_name,
                        repo_id=full_repo_name,
                        repo_type="model",
                        token=token,  # Explicitly pass token
                    )
                    print(f"✅ Uploaded {file_name}")

            # Create and upload model card
            model_card_content = f"""
    # AI Attack Detector

    This model is fine-tuned to detect AI attack queries (prompt injection, etc.) vs benign queries.

    ## Model Details
    - Base model: {self.config.BASE_MODEL}
    - Task: Binary classification (attack vs benign)
    - Framework: PyTorch + Sentence Transformers

    ## Usage

    ```python
    from ai_attack_detector import AIAttackDetector

    detector = AIAttackDetector.from_pretrained("{full_repo_name}")
    result = detector.predict("Your query here")
    print(f"Is attack: {{result['is_attack']}}")
    print(f"Confidence: {{result['confidence']:.4f}}")
    ```
    """

            print("Uploading README.md...")
            api.upload_file(
                path_or_fileobj=model_card_content.encode(),
                path_in_repo="README.md",
                repo_id=full_repo_name,
                repo_type="model",
                token=token,  # Explicitly pass token
            )
            print("✅ Uploaded README.md")

            print(
                f"🎉 Model successfully pushed to: https://huggingface.co/{full_repo_name}"
            )

        except Exception as e:
            print(f"❌ Error during upload: {e}")
            raise

        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print("🧹 Cleaned up temporary files")

        return full_repo_name
