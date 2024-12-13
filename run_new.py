import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from captum.attr import LayerIntegratedGradients
import numpy as np
from typing import List, Tuple, Dict
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

    '''
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

        # 1. Local Coherence Score
        coherence_scores = []
        for attrs in normalized_attrs.values():
            local_coherence = np.mean(np.abs(attrs[1:] - attrs[:-1]))
            coherence_scores.append(local_coherence)
        scores["local_coherence"] = np.mean(coherence_scores)

        # 2. Spatial Clustering Score
        def compute_spatial_clustering(attrs_dict, window_size=3):
            """
            Compute spatial clustering comparing attributions between classes.
            Higher score means more discrete separation between classes.
            """
            classes = list(attrs_dict.keys())
            n_tokens = len(next(iter(attrs_dict.values())))

            dominance_scores = []
            for i in range(n_tokens - window_size + 1):
                window_scores = {}
                for class_idx in classes:
                    window = attrs_dict[class_idx][i : i + window_size]
                    window_scores[class_idx] = np.mean(np.abs(window))

                sorted_scores = sorted(window_scores.values(), reverse=True)
                if len(sorted_scores) > 1:
                    dominance = (sorted_scores[0] - sorted_scores[1]) / (
                        sum(sorted_scores) + 1e-10
                    )
                    dominance_scores.append(dominance)

            return np.mean(dominance_scores)

        scores["spatial_clustering"] = compute_spatial_clustering(normalized_attrs)

        # 3. Attribution Entropy
        def compute_attribution_entropy(attrs_dict, window_size=3):
            """
            Compute entropy of class mixtures across the text.
            Lower entropy indicates more discrete separation.
            """
            n_tokens = len(next(iter(attrs_dict.values())))
            n_classes = len(attrs_dict)

            entropies = []
            for i in range(n_tokens - window_size + 1):
                window_props = []
                total_attr = 0
                for attrs in attrs_dict.values():
                    window = np.abs(attrs[i : i + window_size])
                    window_props.append(np.sum(window))
                    total_attr += np.sum(window)

                if total_attr > 0:
                    window_props = np.array(window_props) / total_attr
                    probs = window_props[window_props > 0]
                    window_entropy = -np.sum(probs * np.log2(probs))
                    max_entropy = np.log2(n_classes)
                    normalized_entropy = window_entropy / max_entropy
                    entropies.append(normalized_entropy)

            return np.mean(entropies)

        scores["attribution_entropy"] = compute_attribution_entropy(normalized_attrs)

        # 4. Pure vs Mixed Segments Ratio
        def compute_segment_ratio(attrs_dict, window_size=3):
            """
            Compute ratio of pure to mixed segments using sliding windows.
            """
            n_tokens = len(next(iter(attrs_dict.values())))
            pure_windows = 0
            mixed_windows = 0

            for i in range(n_tokens - window_size + 1):
                window_avgs = {}
                for class_idx, attrs in attrs_dict.items():
                    window = np.abs(attrs[i : i + window_size])
                    window_avgs[class_idx] = np.mean(window)

                sorted_attrs = sorted(window_avgs.values(), reverse=True)

                if len(sorted_attrs) > 1:
                    dominance_ratio = sorted_attrs[0] / (sorted_attrs[1] + 1e-10)
                    if dominance_ratio > 1.5:
                        pure_windows += 1
                    else:
                        mixed_windows += 1

            return pure_windows / (mixed_windows + 1e-10)

        scores["pure_mixed_ratio"] = compute_segment_ratio(normalized_attrs)

        # 5. Class Dominance Alternation (improved version)
        def compute_dominance_alternation(
            attrs_dict, window_size=3, significance_threshold=1.2
        ):
            """
            Compute meaningful changes in class dominance across the text.

            Args:
                attrs_dict: Dictionary of attributions per class
                window_size: Size of sliding window
                significance_threshold: How much stronger a class needs to be to be considered dominant

            Returns:
                Score between 0 and 1, where:
                - Higher score = more alternation between clearly dominant classes
                - Lower score = either stable dominance or no clear dominance
            """
            n_tokens = len(next(iter(attrs_dict.values())))
            window_dominance = []

            # For each window, find dominant class if any
            for i in range(n_tokens - window_size + 1):
                # Get average attribution for each class in window
                window_avgs = {}
                for class_idx, attrs in attrs_dict.items():
                    window = np.abs(attrs[i : i + window_size])
                    window_avgs[class_idx] = np.mean(window)

                # Sort classes by attribution
                sorted_classes = sorted(
                    window_avgs.items(), key=lambda x: x[1], reverse=True
                )

                # Check if highest is significantly stronger than second
                if len(sorted_classes) > 1:
                    ratio = sorted_classes[0][1] / (sorted_classes[1][1] + 1e-10)
                    if ratio > significance_threshold:
                        window_dominance.append(
                            sorted_classes[0][0]
                        )  # Store dominant class
                    else:
                        window_dominance.append(None)  # No clear dominance

            # Count significant changes in dominance
            significant_changes = 0
            prev_dominant = window_dominance[0]

            for curr_dominant in window_dominance[1:]:
                if curr_dominant is not None and prev_dominant is not None:
                    if curr_dominant != prev_dominant:
                        significant_changes += 1
                prev_dominant = curr_dominant

            # Normalize by number of possible changes
            possible_changes = len(window_dominance) - 1
            return significant_changes / possible_changes if possible_changes > 0 else 0

        scores["dominance_alternation"] = compute_dominance_alternation(
            normalized_attrs
        )

        # Compute final discreteness score
        scores["overall_discreteness"] = (
            0.25 * scores["local_coherence"]
            + 0.25 * scores["spatial_clustering"]
            + 0.25 * (1 - scores["attribution_entropy"])
            + 0.15 * scores["pure_mixed_ratio"]
            + 0.10 * scores["dominance_alternation"]
        )

        return scores, attributions, tokens, true_positives
    '''

    def analyze_hybridity(
        self, text: str, true_classes: List[int]
    ) -> Tuple[Dict[str, float], Dict[int, torch.Tensor], List[str], List[int]]:
        """
        Analyze hybridity type using blockiness metric.
        Returns:
            - Dictionary of scores
            - Token attributions
            - Tokens
            - True positive classes
        """
        result = self.compute_token_attributions(text, true_classes)
        if result is None:
            return None

        attributions, tokens, true_positives = result
        scores = {}

        def normalize_attributions(attrs):
            """Normalize attribution values."""
            attrs = attrs - attrs.mean()
            std = attrs.std()
            if std > 0:
                attrs = attrs / std
            return attrs

        def get_overall_blockiness(
            attrs_class1, attrs_class2, tokens, threshold=0.1, pooling="mean"
        ):
            """
            Calculate blockiness by first combining subtokens into words.

            Args:
                attrs_class1, attrs_class2: Attribution arrays for two classes
                tokens: List of tokens from XLM-RoBERTa tokenizer
                threshold: Threshold for considering changes significant
                pooling: 'mean' or 'max' for how to combine subword attributions
            """
            # Normalize attributions
            norm_attrs1 = normalize_attributions(torch.tensor(attrs_class1))
            norm_attrs2 = normalize_attributions(torch.tensor(attrs_class2))

            abs_attrs1 = np.abs(norm_attrs1.numpy())
            abs_attrs2 = np.abs(norm_attrs2.numpy())

            # Combine subtokens into words
            word_attrs1 = []
            word_attrs2 = []
            current_word_attr1 = []
            current_word_attr2 = []

            for i, token in enumerate(tokens):
                # XLM-RoBERTa uses '▁' at the start of new words
                if token.startswith("▁") and current_word_attr1:  # End of previous word
                    if pooling == "mean":
                        word_attrs1.append(np.mean(current_word_attr1))
                        word_attrs2.append(np.mean(current_word_attr2))
                    else:  # max pooling
                        word_attrs1.append(np.max(current_word_attr1))
                        word_attrs2.append(np.max(current_word_attr2))
                    current_word_attr1 = []
                    current_word_attr2 = []

                # Remove '▁' before checking if it's a special token
                clean_token = token.lstrip("▁")
                if not clean_token.startswith("<") and not clean_token.endswith(
                    ">"
                ):  # Skip special tokens
                    current_word_attr1.append(abs_attrs1[i])
                    current_word_attr2.append(abs_attrs2[i])

            # Handle last word if needed
            if current_word_attr1:
                if pooling == "mean":
                    word_attrs1.append(np.mean(current_word_attr1))
                    word_attrs2.append(np.mean(current_word_attr2))
                else:  # max pooling
                    word_attrs1.append(np.max(current_word_attr1))
                    word_attrs2.append(np.max(current_word_attr2))

            # Calculate differences at word level
            differences = np.abs(np.array(word_attrs1) - np.array(word_attrs2))
            changes = np.abs(np.diff(differences)) > threshold

            return 1 - (np.sum(changes) / (len(differences) - 1))

        # Convert attributions to numpy arrays
        attr_arrays = {
            class_idx: attrs.numpy() for class_idx, attrs in attributions.items()
        }

        # Calculate blockiness for each pair of classes
        blockiness_scores = []
        class_pairs = list(itertools.combinations(attr_arrays.keys(), 2))

        for class1, class2 in class_pairs:
            blockiness = get_overall_blockiness(
                attr_arrays[class1], attr_arrays[class2]
            )
            blockiness_scores.append(blockiness)

        # Average blockiness across all pairs
        scores["blockiness"] = np.mean(blockiness_scores)

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

    for idx, row in tqdm(df.iloc[:10000].iterrows(), desc="Processing texts"):
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
