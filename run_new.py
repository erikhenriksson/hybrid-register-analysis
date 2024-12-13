import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from captum.attr import LayerIntegratedGradients
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm


class RegisterHybridityAnalyzer:
    def __init__(
        self,
        model_name: str = "TurkuNLP/web-register-classification-multilingual",
        threshold: float = 0.4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        self.threshold = threshold

        # Get first 9 main classes and create label mapping
        self.id2label = {
            i: label for i, label in list(self.model.config.id2label.items())[:9]
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        print("Using labels:", self.id2label)

    def forward_func(self, inputs, attention_mask):
        """Forward function for integrated gradients."""
        return self.model(inputs, attention_mask=attention_mask).logits[:, :9]

    def get_predictions(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        """Get prediction probabilities and positive classes."""
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[:, :9]
            probs = torch.sigmoid(logits)

        positive_classes = (probs[0] > self.threshold).nonzero().flatten().tolist()
        return probs[0], positive_classes

    def get_true_classes(self, label_str: str) -> List[int]:
        """Convert space-separated label string to list of class indices."""
        true_labels = label_str.split()
        class_indices = []

        for label in true_labels:
            if label in self.label2id:  # Only consider first 9 main classes
                class_indices.append(self.label2id[label])
        return class_indices

    def compute_token_attributions(
        self, text: str, true_classes: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Compute integrated gradients attributions for each class."""
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get predictions and find true positives
        probs, predicted_classes = self.get_predictions(text)
        true_positives = list(set(true_classes) & set(predicted_classes))

        # If not multiple true positives, return None
        if len(true_positives) <= 1:
            return None

        # Create baseline (pad token) input
        baseline = torch.ones_like(input_ids) * self.tokenizer.pad_token_id

        # Initialize LayerIntegratedGradients
        lig = LayerIntegratedGradients(
            self.forward_func, self.model.roberta.embeddings.word_embeddings
        )

        # Store attributions for each true positive class
        class_attributions = {}
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Calculate attributions for each true positive class
        for class_idx in tqdm(true_positives, desc="Computing attributions"):
            attributions = lig.attribute(
                inputs=input_ids,
                baselines=baseline,
                target=int(class_idx),
                additional_forward_args=(attention_mask,),
                n_steps=50,
                internal_batch_size=10,
                # return_convergence_delta=True,
            )

            # Store attribution scores (summed across embedding dimension)
            attr_scores = attributions.sum(dim=-1).squeeze(0)
            class_attributions[class_idx] = attr_scores.detach().cpu()

        return class_attributions, token_list, true_positives

    def analyze_hybridity(
        self, text: str, true_classes: List[int]
    ) -> Tuple[Dict[str, float], Dict[int, torch.Tensor], List[str], List[int]]:
        """
        Analyze hybridity type using multiple metrics.
        Returns:
            - Dictionary of scores (various metrics)
            - Token attributions
            - Tokens
            - True positive classes
        """
        result = self.compute_token_attributions(text, true_classes)
        if result is None:
            return None

        attributions, tokens, true_positives = result
        scores = {}

        # Normalize attributions for each class
        normalized_attrs = {}
        for class_idx, attrs in attributions.items():
            attrs = attrs.numpy()  # Convert to numpy for easier computation
            attrs = (attrs - attrs.mean()) / (attrs.std() + 1e-10)
            normalized_attrs[class_idx] = attrs

        # 1. Local Coherence Score (original)
        coherence_scores = []
        for attrs in normalized_attrs.values():
            local_coherence = np.mean(np.abs(attrs[1:] - attrs[:-1]))
            coherence_scores.append(local_coherence)
        scores["local_coherence"] = np.mean(coherence_scores)

        # 2. Spatial Clustering Score
        def compute_spatial_clustering(attrs, window_size=3):
            """Compute how much similar attribution values cluster together."""
            clustering_score = 0
            for i in range(len(attrs) - window_size):
                window = attrs[i : i + window_size]
                clustering_score += np.var(window)  # Lower variance = more clustering
            return clustering_score / (len(attrs) - window_size)

        clustering_scores = []
        for attrs in normalized_attrs.values():
            clustering_scores.append(compute_spatial_clustering(attrs))
        scores["spatial_clustering"] = np.mean(clustering_scores)

        # 3. Attribution Entropy
        def compute_entropy(attrs, bins=10):
            """Compute entropy of attribution distribution."""
            hist, _ = np.histogram(attrs, bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zero probabilities
            return -np.sum(hist * np.log2(hist))

        entropy_scores = []
        for attrs in normalized_attrs.values():
            entropy_scores.append(compute_entropy(attrs))
        scores["attribution_entropy"] = np.mean(entropy_scores)

        # 4. Pure vs Mixed Segments Ratio
        def compute_segment_ratio(attrs_dict, threshold=0.5):
            """Compute ratio of pure to mixed segments."""
            n_tokens = len(next(iter(attrs_dict.values())))
            pure_segments = 0
            mixed_segments = 0

            for i in range(n_tokens):
                token_attrs = [abs(attrs[i]) for attrs in attrs_dict.values()]
                max_attr = max(token_attrs)
                if max_attr > threshold:
                    # Check if one class strongly dominates
                    if max_attr > 2 * sum(
                        sorted(token_attrs)[:-1]
                    ):  # One class has more attribution than all others combined
                        pure_segments += 1
                    else:
                        mixed_segments += 1

            return pure_segments / (mixed_segments + 1e-10)  # Avoid division by zero

        scores["pure_mixed_ratio"] = compute_segment_ratio(normalized_attrs)

        # 5. Class Dominance Alternation
        def compute_dominance_alternation(attrs_dict):
            """Compute how often the dominant class changes across tokens."""
            n_tokens = len(next(iter(attrs_dict.values())))
            dominant_classes = []

            for i in range(n_tokens):
                token_attrs = {
                    class_idx: abs(attrs[i]) for class_idx, attrs in attrs_dict.items()
                }
                dominant_class = max(token_attrs.items(), key=lambda x: x[1])[0]
                dominant_classes.append(dominant_class)

            # Count changes in dominance
            changes = sum(
                1
                for i in range(len(dominant_classes) - 1)
                if dominant_classes[i] != dominant_classes[i + 1]
            )
            return changes / (n_tokens - 1)

        scores["dominance_alternation"] = compute_dominance_alternation(
            normalized_attrs
        )

        # Compute final discreteness score (weighted combination of all metrics)
        # Higher score indicates more discrete separation
        scores["overall_discreteness"] = (
            0.3 * scores["local_coherence"]
            + 0.2 * scores["spatial_clustering"]
            + 0.2
            * (
                1 - scores["attribution_entropy"]
            )  # Lower entropy indicates more discreteness
            + 0.15 * scores["pure_mixed_ratio"]
            + 0.15 * scores["dominance_alternation"]
        )

        return scores, attributions, tokens, true_positives


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv(
        "fi.tsv",
        sep="\t",
        header=None,
        names=["register", "text"],
        na_values="",
        keep_default_na=False,
    )

    # Initialize analyzer
    analyzer = RegisterHybridityAnalyzer()

    # Analyze texts
    print(f"Analyzing {len(df)} texts...")
    results = []

    for idx, row in tqdm(df.iloc[:100].iterrows(), desc="Processing texts"):
        true_classes = analyzer.get_true_classes(row["register"])

        # Skip if not multiple true classes in main classes
        if len(true_classes) <= 1:
            continue

        analysis = analyzer.analyze_hybridity(row["text"], true_classes)
        if analysis is not None:
            score, attributions, tokens, true_positives = analysis
            print(f"\nText {idx}:")
            print(f"Text (truncated): {row['text'][:100]}...")
            print(f"True classes: {[analyzer.id2label[c] for c in true_classes]}")
            print(f"True positives: {[analyzer.id2label[c] for c in true_positives]}")
            print(f"Discreteness scores: {score}")
