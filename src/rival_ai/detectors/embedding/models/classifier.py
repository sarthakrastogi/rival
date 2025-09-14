# classifier.py

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from ....config import Config


class AttackClassifier(nn.Module):
    def __init__(
        self,
        base_model_name=None,
        multiclass=False,
        num_classes=None,
        use_contrastive=False,
        is_sentence_transformer=None,
    ):
        super(AttackClassifier, self).__init__()
        self.config = Config()
        self.multiclass = multiclass
        self.use_contrastive = use_contrastive

        if base_model_name is None:
            base_model_name = self.config.BASE_MODEL

        # Determine if this is a sentence transformer or BERT-style model
        if is_sentence_transformer is None:
            self.is_sentence_transformer = self._is_sentence_transformer_model(
                base_model_name
            )
        else:
            self.is_sentence_transformer = is_sentence_transformer

        # Initialize the appropriate model type
        if self.is_sentence_transformer:
            self.sentence_transformer = SentenceTransformer(base_model_name)
            self.tokenizer = None
            self.bert_model = None
            # Get the embedding dimension
            self.embedding_dim = (
                self.sentence_transformer.get_sentence_embedding_dimension()
            )
        else:
            # BERT-style model (like ModernBERT)
            self.sentence_transformer = None
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.bert_model = AutoModel.from_pretrained(base_model_name)
            # Get the embedding dimension from the model config
            self.embedding_dim = self.bert_model.config.hidden_size

        # Determine number of output classes
        if multiclass:
            if num_classes is None:
                # Try to get from config, default to common multi-class scenario
                self.num_classes = getattr(self.config, "NUM_CLASSES", 3)
            else:
                self.num_classes = num_classes
        else:
            self.num_classes = 1  # Binary classification

        # Classification head
        if multiclass:
            # Multi-class classifier (no sigmoid, use softmax in loss)
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes),
            )
        else:
            # Binary classifier
            if use_contrastive:
                # For contrastive loss, don't use sigmoid - will be applied later
                self.classifier = nn.Sequential(
                    nn.Linear(self.embedding_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),  # No sigmoid here
                )
            else:
                # Standard binary classifier with sigmoid
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

    def forward(self, texts):
        # Handle both list of strings and batch dictionary format
        if isinstance(texts, (list, tuple)) and len(texts) > 0:
            if isinstance(texts[0], dict):
                # Handle dictionary format from DataLoader
                text_list = [item["text"] for item in texts]
            elif isinstance(texts[0], str):
                # Handle list of strings
                text_list = texts
            else:
                # Handle other formats
                text_list = [str(text) for text in texts]
        elif isinstance(texts, dict):
            # Handle single dictionary
            text_list = [texts["text"]]
        else:
            # Handle single string or other formats
            text_list = [str(texts)]

        # Get sentence embeddings based on model type
        if self.is_sentence_transformer:
            # Use sentence transformer
            with torch.no_grad():
                embeddings = self.sentence_transformer.encode(
                    text_list, convert_to_tensor=True, device=self.config.DEVICE
                )
        else:
            # Use BERT-style model (like ModernBERT)
            # Tokenize the input
            inputs = self.tokenizer(
                text_list,
                padding=True,
                truncation=True,
                max_length=self.config.MAX_SEQ_LENGTH,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}

            # Get embeddings from BERT model
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use mean pooling of the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)

        # Classification
        outputs = self.classifier(embeddings)

        if self.multiclass:
            return outputs  # Return raw logits for multi-class
        else:
            if self.use_contrastive:
                return outputs.squeeze()  # Return raw logits for contrastive loss
            else:
                return outputs.squeeze()  # Return probabilities for binary

    def forward_embeddings(self, embeddings):
        """Forward pass using pre-computed embeddings."""
        outputs = self.classifier(embeddings)

        if self.multiclass:
            return outputs
        else:
            if self.use_contrastive:
                return outputs.squeeze()  # Raw logits
            else:
                return outputs.squeeze()  # Probabilities

    def get_embeddings(self, texts):
        """Get embeddings for texts (useful for contrastive loss)."""
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, torch.Tensor):
            raise ValueError(
                "get_embeddings received tensor input instead of text strings. "
                "Please provide raw text strings."
            )

        if self.is_sentence_transformer:
            return self.sentence_transformer.encode(
                texts, convert_to_tensor=True, device=self.config.DEVICE
            )
        else:
            # Use BERT-style model
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.MAX_SEQ_LENGTH,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings

    def encode_texts(self, texts):
        """Get embeddings for texts."""
        return self.get_embeddings(texts)

    def predict_with_embeddings(self, embeddings, apply_sigmoid=True):
        """Make predictions using pre-computed embeddings."""
        with torch.no_grad():
            outputs = self.classifier(embeddings)

            if self.multiclass:
                return torch.softmax(outputs, dim=1)
            else:
                if apply_sigmoid or not self.use_contrastive:
                    return torch.sigmoid(outputs.squeeze())
                else:
                    return outputs.squeeze()  # Raw logits
