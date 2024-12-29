import numpy as np
from math import log2
import torch
import torch.nn.functional as F
import pandas as pd
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import spacy
from tqdm import tqdm
import sys


# Scoring parameters (weights for each component)
ALPHA = 1.0  # Weight for average entropy
BETA = 0.0  # Weight for average KL divergence
GAMMA = 0.0  # Weight for mutual information
DELTA = 0.0  # Weight for average variance
LAMBDA = 0.05  # Tunable parameter for over-segmentation
MU = 0.05  # Tunable parameter for short segments

labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

labels_all = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]


def get_group_probabilities(prob_list):
    group_probs = []

    for label, children in labels_structure.items():
        # Get index of parent label
        parent_idx = labels_all.index(label)
        parent_prob = prob_list[parent_idx]

        if not children:  # Labels like MT, LY, ID
            group_probs.append(parent_prob)
        else:  # Labels like NA, IN, etc
            # Get indices and probabilities of children
            children_probs = [prob_list[labels_all.index(child)] for child in children]
            # Take maximum probability in the group
            group_probs.append(max(parent_prob, *children_probs))

    return group_probs


def calculate_entropy(probabilities):
    """
    Calculate the entropy for a single set of probabilities (multilabel).
    """
    eps = 1e-12  # To avoid log(0)
    probabilities = np.clip(
        probabilities, eps, 1 - eps
    )  # Clip probabilities to avoid numerical issues
    entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)
    return entropy.mean()


def calculate_multilabel_entropy(probabilities):
    """
    Calculate the entropy for multilabel predictions where each position
    represents an independent binary decision.
    """
    eps = 1e-12  # To avoid log(0)
    probabilities = np.clip(probabilities, eps, 1 - eps)

    # Calculate binary entropy for each position
    entropy = -(
        probabilities * np.log(probabilities)
        + (1 - probabilities) * np.log(1 - probabilities)
    )

    return entropy.mean()


def calculate_kl_divergence(prob_a, prob_b):
    """
    Calculate KL divergence between two probability distributions (multilabel).
    """
    eps = 1e-12
    prob_a = np.clip(prob_a, eps, 1 - eps)
    prob_b = np.clip(prob_b, eps, 1 - eps)
    kl_div = np.sum(prob_a * np.log(prob_a / prob_b), axis=-1)
    return kl_div.mean()


def calculate_mutual_information(global_probs, block_probs):
    """
    Calculate mutual information for a block given the global probabilities and the block probabilities.
    """
    global_entropy = calculate_multilabel_entropy(global_probs)
    block_entropy = calculate_multilabel_entropy(block_probs)
    return global_entropy - block_entropy


def calculate_variance(probabilities):
    """
    Calculate the variance of probabilities within a block.
    """
    return np.var(probabilities, axis=0).mean()


def score_partition(partition_predictions, global_predictions, n, partition_texts):
    """
    Compute the score for a given partition based on entropy, KL divergence, mutual information,
    variance, and penalties for over-segmentation and short segments.
    """
    num_blocks = len(partition_predictions)

    # Group probs
    partition_predictions = [get_group_probabilities(x) for x in partition_predictions]

    # Calculate base metrics
    avg_entropy = np.mean(
        [calculate_multilabel_entropy(block) for block in partition_predictions]
    )
    avg_kl_div = 0.0

    for i in range(num_blocks - 1):
        avg_kl_div += calculate_kl_divergence(
            partition_predictions[i], partition_predictions[i + 1]
        )
    if num_blocks > 1:
        avg_kl_div /= num_blocks - 1

    mutual_info = np.mean(
        [
            calculate_mutual_information(global_predictions, block)
            for block in partition_predictions
        ]
    )
    avg_variance = np.mean(
        [calculate_variance(block) for block in partition_predictions]
    )

    # Calculate penalties

    # 1. Over-segmentation penalty
    # We need to pass the total number of sentences to this function
    # Let's assume n is passed as an additional parameter
    oversegmentation_penalty = num_blocks / n

    # 2. Short segment penalty
    # We need the original texts to calculate this
    # Let's assume partition_texts is passed as an additional parameter

    avg_length = np.mean([len(text.split()) for text in partition_texts])
    short_penalties = [
        max(0, 1 - len(text.split()) / (avg_length)) ** 2 for text in partition_texts
    ]
    short_segment_penalty = np.mean(short_penalties)

    # Combine all components
    score = (
        ALPHA * -avg_entropy
        + BETA * avg_kl_div
        + GAMMA * mutual_info
        - DELTA * avg_variance
        - LAMBDA * oversegmentation_penalty  # Add penalties
        - MU * short_segment_penalty
    )
    return score


# Create the child to parent index mapping
child_to_parent_idx = {}

# Iterate through the structure
for parent, children in labels_structure.items():
    parent_idx = labels_all.index(parent)  # Get parent's index
    for child in children:
        child_idx = labels_all.index(child)  # Get child's index
        child_to_parent_idx[child_idx] = parent_idx

# Get main categories (keys from labels_structure)
labels_main = list(labels_structure.keys())

# Map children to parents
labels_parents = {}
for parent, children in labels_structure.items():
    for child in children:
        labels_parents[child] = parent

# Load register classification model
model_name = "TurkuNLP/web-register-classification-multilingual"
tokenizer_name = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_tokens = 512
min_words = 30
max_segments = 20
model = model.to(device)
model.eval()

# Load E5 embedding model
embed_model_name = "intfloat/multilingual-e5-large"
embed_model = AutoModel.from_pretrained(embed_model_name)
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = embed_model.to(device)
embed_model.eval()

# Load spaCy
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def predict_batch(texts, batch_size=32):
    """Predict probabilities for a batch of texts."""
    all_probs = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        # Move input tensors to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Move predictions back to CPU and convert to numpy
        batch_probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()

        # Round to three decimals
        batch_probs = [[round(prob, 3) for prob in probs] for probs in batch_probs]
        all_probs.extend(batch_probs)

    return all_probs

"""
def get_embeddings_batch(texts, batch_size=32):
    """Get embeddings for texts in batches using E5 model."""
    all_embeddings = []

    # Add prefix for E5
    texts = ["passage: " + text for text in texts]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize
        inputs = embed_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Get embeddings
        with torch.no_grad():
            outputs = embed_model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)
"""

def get_embeddings_batch(texts, batch_size=32):
    """Get embeddings for texts in batches using RoBERTa model."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize batch using your existing RoBERTa tokenizer
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Get embeddings
        with torch.no_grad():
            # We need to modify the forward pass to get hidden states
            outputs = model(
                **inputs,
                output_hidden_states=True  # Request hidden states
            )
            
            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            
            # Extract the [CLS] token embeddings (first token of each sequence)
            cls_embeddings = last_hidden_state[:, 0, :]

        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

def combine_short_sentences(
    sentences, initial_min_words=min_words, max_segments=max_segments
):
    min_words = initial_min_words

    def count_words(sentence):
        return len(sentence.split())

    while 1:
        result = []
        buffer = ""

        for i, sentence in enumerate(sentences):
            if count_words(sentence) >= min_words:
                if buffer:
                    result.append(buffer.strip())
                    buffer = ""
                result.append(sentence)
            else:
                buffer += (buffer and " ") + sentence

                # If the buffer reaches min_words, finalize it
                if count_words(buffer) >= min_words:
                    result.append(buffer.strip())
                    buffer = ""

        # Handle leftover buffer
        if buffer:
            result.append(buffer.strip())

        # Final pass: Ensure no sentences in the result are below min_words
        i = 0
        while i < len(result):
            if count_words(result[i]) < min_words:
                if i < len(result) - 1:  # Merge with the next sentence
                    result[i + 1] = result[i] + " " + result[i + 1]
                    result.pop(i)
                elif i > 0:  # Merge with the previous sentence if it's the last one
                    result[i - 1] += " " + result[i]
                    result.pop(i)
                else:  # Single short sentence case
                    break
            else:
                i += 1
        if len(result) <= max_segments:
            return result
        min_words += 1


def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def truncate_text_to_tokens(text):
    """Truncate text to fit within model's token limit"""
    tokens = tokenizer(text, truncation=True, max_length=max_tokens)
    truncated_text = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
    return truncated_text


def generate_unique_subsequences(sentences):
    """Generate all unique contiguous subsequences of sentences."""
    n = len(sentences)
    result = []
    for start in range(n):
        for length in range(1, n - start + 1):
            subsequence = sentences[start : start + length]
            result.append(subsequence)
    return result


def generate_partitionings_with_entropy(sentences):
    """Generate partitionings considering both entropy and semantic differences."""
    subsequences = generate_unique_subsequences(sentences)
    print("Subsequences: ", len(subsequences))

    # Get all unique texts
    texts = [" ".join(subseq) for subseq in subsequences]

    # Get predictions and embeddings for all unique segments
    predictions = predict_batch(texts)
    embeddings = get_embeddings_batch(texts)

    n = len(sentences)

    def build_partitions(current_partition, start_idx):
        if start_idx == n:
            results.append(current_partition[:])
            return

        for idx, subsequence in enumerate(subsequences):
            if subsequence[0] == sentences[start_idx]:
                if start_idx + len(subsequence) <= n:
                    current_partition.append(idx)
                    build_partitions(current_partition, start_idx + len(subsequence))
                    current_partition.pop()

    results = []
    build_partitions([], 0)

    print("Sentences:", len(sentences))
    print("Partitions:", len(results))

    # Find partition with best combined score
    # Initialize variables
    max_score = -float("inf")
    best_partition_indices = None

    for partition_indices in results:
        # Get the predictions from indices
        partition_predictions = [predictions[idx] for idx in partition_indices]

        # Assume global predictions are the average of all predictions
        global_predictions = np.mean(predictions, axis=0)

        # Compute the score for this partition
        score = score_partition(
            partition_predictions,
            global_predictions,
            len(sentences),  # total number of sentences
            [
                " ".join(subsequences[idx]) for idx in partition_indices
            ],  # texts for length calculation
        )

        if score > max_score:
            max_score = score
            best_partition_indices = partition_indices

    # Convert best partition indices back to text
    best_partition_text = [
        [s for s in subsequences[idx]] for idx in best_partition_indices
    ]
    best_partition_probs = [
        [round(float(y), 3) for y in predictions[idx]] for idx in best_partition_indices
    ]

    best_partition_embeddings = embeddings[best_partition_indices]

    return (
        best_partition_text,
        best_partition_probs,
        best_partition_embeddings,
        round(max_score, 3),
    )


def get_dominant_registers(probs, threshold=0.4):
    """Get names of registers that pass the threshold."""
    dominant = [labels_all[i] for i, p in enumerate(probs) if p >= threshold]
    return dominant if dominant else ["None"]


def process_tsv_file(input_file_path, output_file_path):
    """Process texts from a TSV file and generate predictions, saving results to JSONL."""
    # Read TSV file
    df = pd.read_csv(input_file_path, sep="\t", header=None)

    # Process each text and write results to JSONL
    for idx, text in enumerate(df[1]):
        truncated_text = truncate_text_to_tokens(text)

        # Split text into sentences
        sentences = split_into_sentences(truncated_text)

        # Combine short sentences
        combined_sentences = combine_short_sentences(sentences)

        # Generate partitions and predictions
        best_partition, partition_probs, best_partition_embeddings, max_score = (
            generate_partitionings_with_entropy(combined_sentences)
        )

        # Create result dictionary with proper type conversion
        result = {
            "best_partition": best_partition,
            "partition_probs": [
                [float(prob) for prob in probs] for probs in partition_probs
            ],
            "best_partition_embeddings": best_partition_embeddings.tolist(),
            "max_score": float(max_score),
        }

        # Write to JSONL file
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Print progress and results
        print(f"\nProcessed text {idx + 1}/{len(df)}")
        print("Predictions for each part:")
        for part, probs in zip(best_partition, partition_probs):
            dominant_registers = get_dominant_registers(probs)
            print(f"Dominant registers: {', '.join(dominant_registers)}")
            print(f"Text: {' '.join(part)}")
            print(f"Probabilities: {probs}\n")
        print("Maximum combined score:", max_score)
        print("-" * 80)


# Example usage
if __name__ == "__main__":
    # Get file name from sys argv
    input_file = sys.argv[1]

    # Output file, add _results before extension
    output_file = input_file.replace(".tsv", "_results.jsonl")

    process_tsv_file(input_file, output_file)
