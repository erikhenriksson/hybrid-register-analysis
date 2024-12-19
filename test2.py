import numpy as np
from math import log2
import torch
import torch.nn.functional as F
import pandas as pd
import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import spacy
from tqdm import tqdm
import sys

# Load register classification model
model_name = "TurkuNLP/web-register-classification-multilingual"
tokenizer_name = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_tokens = 512
min_words = 30
max_groups = 20
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

        # Round to three decimals and take first 9 probabilities
        batch_probs = [[round(prob, 3) for prob in probs[:9]] for probs in batch_probs]
        all_probs.extend(batch_probs)

    return all_probs


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


def calculate_semantic_difference(embeddings, partition_indices):
    """Calculate average semantic difference between adjacent segments."""
    if len(partition_indices) <= 1:
        return 0.0

    partition_embeddings = embeddings[partition_indices]
    differences = []

    for i in range(len(partition_embeddings) - 1):
        similarity = F.cosine_similarity(
            partition_embeddings[i : i + 1], partition_embeddings[i + 1 : i + 2]
        )
        difference = 1 - similarity.item()
        differences.append(difference)

    return sum(differences) / len(differences)


def calculate_entropy(probs):
    """Calculate binary entropy for each probability in multilabel classification."""

    def binary_entropy(p):
        # Handle edge cases to avoid log(0)
        if p <= 0 or p >= 1:
            return 0
        # Calculate entropy for binary probability (p, 1-p)
        return -(p * log2(p) + (1 - p) * log2(1 - p))

    probs = np.array(probs)
    # Calculate binary entropy for each probability
    return sum(binary_entropy(p) for p in probs)


def combine_short_sentences(
    sentences, initial_min_words=min_words, max_groups=max_groups
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
        if len(result) <= max_groups:
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

    print("Partitions: ", len(results))

    # Find partition with best combined score
    max_score = -float("inf")
    best_partition_indices = None

    for partition_indices in results:
        # Get entropy score
        partition_predictions = [predictions[idx] for idx in partition_indices]
        entropies = [calculate_entropy(pred) for pred in partition_predictions]
        entropy_score = sum(entropies) / len(entropies)

        # Get semantic difference score
        semantic_score = calculate_semantic_difference(embeddings, partition_indices)

        # Combine scores with equal weights
        combined_score = -entropy_score + semantic_score

        if combined_score > max_score:
            max_score = combined_score
            best_partition_indices = partition_indices

    # Convert best partition indices back to text
    best_partition_text = [
        [s for s in subsequences[idx]] for idx in best_partition_indices
    ]
    best_partition_probs = [
        [round(float(y), 3) for y in predictions[idx]] for idx in best_partition_indices
    ]

    return best_partition_text, best_partition_probs, round(max_score, 3)


def get_dominant_registers(probs, threshold=0.4):
    """Get names of registers that pass the threshold."""
    REGISTER_NAMES = ["MT", "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]
    dominant = [REGISTER_NAMES[i] for i, p in enumerate(probs) if p >= threshold]
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
        best_partition, partition_probs, max_score = (
            generate_partitionings_with_entropy(combined_sentences)
        )

        # Create result dictionary with proper type conversion
        result = {
            "best_partition": best_partition,
            "partition_probs": [
                [float(prob) for prob in probs] for probs in partition_probs
            ],
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
