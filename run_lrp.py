import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from captum.attr import DeepLift

from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import itertools


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

    def get_true_classes(self, label_str: str) -> List[int]:
        """Convert space-separated label string to list of class indices."""
        true_labels = label_str.split()
        class_indices = []

        for label in true_labels:
            if label in self.label2id:  # Only consider first 9 main classes
                class_indices.append(self.label2id[label])

        return class_indices

    def predict_probs(self, text: str) -> Tuple[torch.Tensor, List[int]]:
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

    def compute_lrp_attributions(
        self, text: str, true_classes: List[int]
    ) -> Optional[Tuple[Dict[int, torch.Tensor], List[str], List[int]]]:
        """
        Compute DeepLift attributions for true positive classes.
        """
        # Tokenize input
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get predictions and find true positives
        probs, predicted_classes = self.predict_probs(text)
        true_positives = list(set(true_classes) & set(predicted_classes))

        # If not multiple true positives, return None
        if len(true_positives) <= 1:
            return None

        # Create a wrapper for the forward function that returns only logits
        def forward_wrapper(input_ids, attention_mask):
            return self.model(input_ids, attention_mask=attention_mask).logits[:, :9]

        # Initialize DeepLift with the wrapper
        deep_lift = DeepLift(forward_wrapper)
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Store attributions for each true positive class
        attributions = {}
        for class_idx in true_positives:
            # Compute DeepLift attributions
            attr = deep_lift.attribute(
                input_ids,
                target=int(class_idx),
                additional_forward_args=(attention_mask,),
                baselines=input_ids * 0 + self.tokenizer.pad_token_id,
            )

            # Get attribution scores and mask padding
            attr_scores = attr.sum(dim=-1)[0]
            mask = attention_mask[0].bool()
            attributions[class_idx] = attr_scores[mask].detach().cpu()

        return attributions, token_list, true_positives

    def combine_subwords(
        self, tokens: List[str], attributions: Dict[int, torch.Tensor]
    ) -> Tuple[Dict[int, np.ndarray], List[str]]:
        """
        Combine subword tokens into words and their corresponding attributions.
        """
        word_attrs = {k: [] for k in attributions.keys()}
        words = []
        current_word = []
        current_attrs = {k: [] for k in attributions.keys()}

        for i, token in enumerate(tokens):
            # XLM-RoBERTa uses '▁' at the start of new words
            if token.startswith("▁") and current_word:
                # Add previous word
                for k in attributions.keys():
                    word_attrs[k].append(np.mean(current_attrs[k]))
                words.append("".join(current_word))
                current_word = []
                current_attrs = {k: [] for k in attributions.keys()}

            # Process current token
            clean_token = token.lstrip("▁")
            if not clean_token.startswith("<") and not clean_token.endswith(">"):
                for k, attrs in attributions.items():
                    current_attrs[k].append(attrs[i].item())
                current_word.append(clean_token)

        # Handle last word
        if current_word:
            for k in attributions.keys():
                word_attrs[k].append(np.mean(current_attrs[k]))
            words.append("".join(current_word))

        # Convert lists to numpy arrays
        word_attrs = {k: np.array(v) for k, v in word_attrs.items()}

        return word_attrs, words

    def analyze_hybridity(
        self, text: str, true_classes: List[int], window_size: int = 3
    ) -> Optional[Dict[str, float]]:
        """
        Analyze register hybridity using LRP attributions.
        Returns various metrics for hybrid analysis.
        """
        # Get LRP attributions
        result = self.compute_lrp_attributions(text, true_classes)
        if result is None:
            return None

        attributions, tokens, true_positives = result

        # Combine subwords into words
        word_attrs, words = self.combine_subwords(tokens, attributions)

        # Initialize scores dictionary
        scores = {}

        # 1. Compute attribution pattern correlations between classes
        correlations = []
        for c1, c2 in itertools.combinations(word_attrs.keys(), 2):
            corr = np.corrcoef(word_attrs[c1], word_attrs[c2])[0, 1]
            correlations.append(corr)
        scores["mean_correlation"] = np.mean(correlations)

        # 2. Compute local variations using sliding windows
        local_variations = []
        for attrs in word_attrs.values():
            for i in range(len(attrs) - window_size + 1):
                window = attrs[i : i + window_size]
                local_variations.append(np.std(window))
        scores["local_variation"] = np.mean(local_variations)

        # 3. Compute dominance patterns
        class_dominance = []
        for i in range(len(words)):
            values = {k: abs(attrs[i]) for k, attrs in word_attrs.items()}
            max_val = max(values.values())
            total = sum(values.values())
            if total > 0:
                dominance = max_val / total
                class_dominance.append(dominance)
        scores["mean_dominance"] = np.mean(class_dominance)

        # Store metadata
        scores["text"] = text
        scores["true_classes"] = [self.id2label[c] for c in true_classes]
        scores["true_positives"] = [self.id2label[c] for c in true_positives]
        scores["n_words"] = len(words)

        return scores


def analyze_corpus(
    analyzer: RegisterHybridityAnalyzer,
    df: pd.DataFrame,
    n_samples: Optional[int] = None,
):
    """
    Analyze a corpus of texts and return results as a DataFrame.
    """
    results = []
    texts = df.iloc[:n_samples] if n_samples else df

    for idx, row in tqdm(texts.iterrows(), desc="Analyzing texts"):
        true_classes = analyzer.get_true_classes(row["register"])

        if len(true_classes) <= 1:
            continue

        scores = analyzer.analyze_hybridity(row["text"], true_classes)
        if scores is not None:
            scores["idx"] = idx
            results.append(scores)

    results_df = pd.DataFrame(results)

    # Print summary statistics
    print("\nAnalysis Summary:")
    print(f"Total texts analyzed: {len(results_df)}")

    for metric in ["mean_correlation", "local_variation", "mean_dominance"]:
        print(f"\n{metric} statistics:")
        print(results_df[metric].describe())

    return results_df


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("fi.tsv", sep="\t", header=None, names=["register", "text"])

    # Initialize analyzer
    analyzer = RegisterHybridityAnalyzer()

    # Analyze corpus
    results_df = analyze_corpus(analyzer, df, n_samples=100)  # Start with 100 samples

    # Save results
    results_df.to_csv("hybrid_analysis_results.csv", index=False)

    # Print some interesting examples
    print("\nMost discrete texts (high local variation, low correlation):")
    discrete = results_df.nlargest(5, "local_variation").nsmallest(
        3, "mean_correlation"
    )
    print(
        discrete[
            [
                "text",
                "true_classes",
                "mean_correlation",
                "local_variation",
                "mean_dominance",
            ]
        ]
    )

    print("\nMost blended texts (low local variation, high correlation):")
    blended = results_df.nsmallest(5, "local_variation").nlargest(3, "mean_correlation")
    print(
        blended[
            [
                "text",
                "true_classes",
                "mean_correlation",
                "local_variation",
                "mean_dominance",
            ]
        ]
    )
