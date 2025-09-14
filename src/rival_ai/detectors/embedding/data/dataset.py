import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from ..config import Config
import json
import os


class AttackDataset(Dataset):
    def __init__(self, texts, labels, multiclass=False):
        self.texts = texts
        self.labels = labels
        self.multiclass = multiclass

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Ensure labels are properly converted
        label = self.labels[idx]
        if isinstance(label, (list, tuple)):
            label = label[0] if len(label) > 0 else 0

        if self.multiclass:
            return {"text": str(self.texts[idx]), "label": int(label)}
        else:
            return {"text": str(self.texts[idx]), "label": float(label)}


class DataLoader:
    def __init__(self, csv_path, multiclass=False, label_mapping_path=None):
        self.csv_path = csv_path
        self.config = Config()
        self.multiclass = multiclass
        self.label_mapping_path = label_mapping_path
        self.label_mapping = None
        self.num_classes = None

        # Load label mapping if provided and multiclass is enabled
        if self.multiclass and self.label_mapping_path:
            self.load_label_mapping()

    def load_label_mapping(self):
        """Load label mapping from JSON file."""
        if os.path.exists(self.label_mapping_path):
            with open(self.label_mapping_path, "r") as f:
                self.label_mapping = json.load(f)
            # Convert string keys to integers if needed
            if isinstance(list(self.label_mapping.keys())[0], str):
                self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
            self.num_classes = len(self.label_mapping)
            print(f"Loaded label mapping: {self.label_mapping}")
            print(f"Number of classes: {self.num_classes}")
        else:
            print(f"Warning: Label mapping file not found at {self.label_mapping_path}")

    def load_data(self):
        """Load data from CSV file with proper parsing."""
        # Read CSV with explicit parsing options
        df = pd.read_csv(
            self.csv_path,
            encoding="utf-8",
            quotechar='"',
            skipinitialspace=True,
            na_values=["", "nan", "NaN", "null", "None"],
        )

        # Clean up column names (remove whitespace)
        df.columns = df.columns.str.strip()

        print(f"Loaded CSV with columns: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        print(f"Label column dtype: {df[self.config.LABEL_COLUMN].dtype}")
        print(f"Sample labels: {df[self.config.LABEL_COLUMN].head().tolist()}")

        # Ensure required columns exist
        if (
            self.config.TEXT_COLUMN not in df.columns
            or self.config.LABEL_COLUMN not in df.columns
        ):
            raise ValueError(
                f"CSV must contain '{self.config.TEXT_COLUMN}' and '{self.config.LABEL_COLUMN}' columns. "
                f"Found columns: {list(df.columns)}"
            )

        # Clean and extract data
        texts = df[self.config.TEXT_COLUMN].astype(str).tolist()

        # Handle labels more carefully
        labels_series = df[self.config.LABEL_COLUMN]

        # Convert labels to numeric, handling any parsing issues
        if labels_series.dtype == "object":
            # Try to convert string representations to numeric
            labels = (
                pd.to_numeric(labels_series, errors="coerce")
                .fillna(0)
                .astype(int)
                .tolist()
            )
        else:
            labels = labels_series.astype(int).tolist()

        # Validate labels
        unique_labels = set(labels)
        print(f"Unique labels found: {unique_labels}")

        if self.multiclass:
            # For multi-class, validate against label mapping if available
            if self.label_mapping:
                expected_labels = set(self.label_mapping.keys())
                if not unique_labels.issubset(expected_labels):
                    unexpected = unique_labels - expected_labels
                    print(f"Warning: Found unexpected labels: {unexpected}")
                    print(f"Expected labels: {expected_labels}")

            # Update num_classes based on actual data if not set from mapping
            if self.num_classes is None:
                self.num_classes = len(unique_labels)

            print(f"Multi-class classification with {self.num_classes} classes")

        else:
            # For binary classification, ensure labels are 0/1
            if not unique_labels.issubset({0, 1}):
                print(f"Warning: Found non-binary labels: {unique_labels}")
                # Convert to binary if needed
                labels = [1 if label > 0 else 0 for label in labels]
                print(f"Converted to binary labels: {set(labels)}")

        return texts, labels

    def create_train_test_split(self, texts, labels):
        """Create train/test split."""
        print(f"Creating train/test split with {len(texts)} samples")
        print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=1 - self.config.TRAIN_TEST_SPLIT,
            random_state=self.config.RANDOM_SEED,
            stratify=labels,
        )

        train_dataset = AttackDataset(train_texts, train_labels, self.multiclass)
        test_dataset = AttackDataset(test_texts, test_labels, self.multiclass)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        return train_dataset, test_dataset

    def get_num_classes(self):
        """Get number of classes for the dataset."""
        return self.num_classes

    def get_label_mapping(self):
        """Get label mapping dictionary."""
        return self.label_mapping
