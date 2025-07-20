import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from ..config import Config


class AttackClassifier(nn.Module):
    def __init__(self, base_model_name=None):
        super(AttackClassifier, self).__init__()
        self.config = Config()

        if base_model_name is None:
            base_model_name = self.config.BASE_MODEL

        self.sentence_transformer = SentenceTransformer(base_model_name)

        # Get the embedding dimension
        self.embedding_dim = (
            self.sentence_transformer.get_sentence_embedding_dimension()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, texts):
        # Get sentence embeddings
        with torch.no_grad():
            embeddings = self.sentence_transformer.encode(
                texts, convert_to_tensor=True, device=self.config.DEVICE
            )

        # Classification
        outputs = self.classifier(embeddings)
        return outputs.squeeze()

    def encode_texts(self, texts):
        """Get embeddings for texts."""
        return self.sentence_transformer.encode(
            texts, convert_to_tensor=True, device=self.config.DEVICE
        )
