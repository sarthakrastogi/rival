import re
import string


class TextPreprocessor:
    def __init__(self, max_length=512):
        self.max_length = max_length

    def clean_text(self, text):
        """Basic text cleaning."""
        if not isinstance(text, str):
            text = str(text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Truncate if too long
        if len(text) > self.max_length:
            text = text[: self.max_length]

        return text.strip()

    def preprocess_batch(self, texts):
        """Preprocess a batch of texts."""
        return [self.clean_text(text) for text in texts]
