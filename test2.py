import numpy as np
from math import log2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import spacy
import sys

# Load spaCy
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Load models
model_name = "TurkuNLP/web-register-classification-multilingual"
tokenizer_name = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Load E5 embedding model
embed_model_name = "intfloat/multilingual-e5-large"
embed_model = AutoModel.from_pretrained(embed_model_name)
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
embed_model = embed_model.to(device)
model.eval()
embed_model.eval()


def split_into_sentences(text):
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def combine_short_sentences(sentences, min_words=20, max_groups=20):
    """Combine short sentences into longer segments."""
    initial_min_words = min_words

    def count_words(sentence):
        return len(sentence.split())

    while True:
        result = []
        buffer = ""

        for sentence in sentences:
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
                if i < len(result) - 1:  # Merge with next sentence
                    result[i + 1] = result[i] + " " + result[i + 1]
                    result.pop(i)
                elif i > 0:  # Merge with previous sentence if last
                    result[i - 1] += " " + result[i]
                    result.pop(i)
                else:  # Single short sentence case
                    break
            else:
                i += 1

        if len(result) <= max_groups:
            return result
        min_words += 1


def predict_batch(texts, batch_size=32):
    """Predict probabilities for a batch of texts."""
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
        batch_probs = [[round(prob, 3) for prob in probs[:9]] for probs in batch_probs]
        all_probs.extend(batch_probs)

    return all_probs


def get_segment_embeddings(texts, batch_size=32):
    """Get E5 embeddings for text segments using mean pooling."""
    all_embeddings = []

    # Process texts in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # E5 models expect prefix "passage: " for text embedding
        batch_texts = [f"passage: {text}" for text in batch_texts]

        # Tokenize batch
        inputs = embed_tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        # Move input tensors to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = embed_model(**inputs)

            # Mean pooling with attention mask
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = embeddings.detach().cpu()

        all_embeddings.append(embeddings)

    # Combine all batches
    return torch.cat(all_embeddings, dim=0)


def calculate_entropy(probs):
    """Calculate binary entropy for each probability in multilabel classification."""

    def binary_entropy(p):
        if p <= 0 or p >= 1:
            return 0
        return -(p * log2(p) + (1 - p) * log2(1 - p))

    probs = np.array(probs)
    return sum(binary_entropy(p) for p in probs)


def get_entropy_score(segments_predictions):
    """Higher is better - means predictions are more discrete"""
    entropies = [calculate_entropy(pred) for pred in segments_predictions]
    return sum(entropies) / len(entropies)


def get_semantic_difference_score(partition_indices, segment_embeddings):
    """Calculate semantic difference score from precomputed embeddings."""
    if len(partition_indices) <= 1:
        return 0

    # Get embeddings for this partition
    partition_embeddings = segment_embeddings[partition_indices]

    # Calculate differences between adjacent segments
    differences = []
    for i in range(len(partition_embeddings) - 1):
        similarity = F.cosine_similarity(
            partition_embeddings[i : i + 1], partition_embeddings[i + 1 : i + 2]
        )
        difference = 1 - similarity.item()
        differences.append(difference)

    return sum(differences) / len(differences)


def is_pareto_dominated(scores, candidate_scores):
    """Return True if candidate_scores is dominated by any existing scores"""
    entropy_score, semantic_score = candidate_scores
    return any(
        existing_entropy >= entropy_score
        and existing_semantic >= semantic_score
        and (existing_entropy > entropy_score or existing_semantic > semantic_score)
        for existing_entropy, existing_semantic in scores
    )


def generate_unique_subsequences(sentences):
    """Generate all unique contiguous subsequences of sentences."""
    n = len(sentences)
    result = []
    for start in range(n):
        for length in range(1, n - start + 1):
            subsequence = sentences[start : start + length]
            result.append(subsequence)
    return result


def generate_partitionings_with_pareto(sentences):
    """Generate partitionings optimizing both entropy and semantic difference."""
    subsequences = generate_unique_subsequences(sentences)
    print("Subsequences: ", len(subsequences))

    # Get predictions and embeddings for all unique subsequences
    texts = [" ".join(subseq) for subseq in subsequences]
    predictions = predict_batch(texts)

    # Get embeddings for all unique subsequences
    embeddings = get_segment_embeddings(texts)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings

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

    # Find Pareto-optimal solutions
    pareto_front = []  # Will store (partition_indices, entropy_score, semantic_score)
    scores = []  # Will store (entropy_score, semantic_score) tuples

    for partition_indices in tqdm(results):
        partition_preds = [predictions[idx] for idx in partition_indices]

        entropy_score = get_entropy_score(partition_preds)
        semantic_score = get_semantic_difference_score(partition_indices, embeddings)

        if not is_pareto_dominated(scores, (entropy_score, semantic_score)):
            # Remove any solutions that this new solution dominates
            scores = [
                (e, s)
                for e, s in scores
                if not is_pareto_dominated([(entropy_score, semantic_score)], (e, s))
            ]
            pareto_front = [
                (p, e, s)
                for p, e, s in pareto_front
                if not is_pareto_dominated([(entropy_score, semantic_score)], (e, s))
            ]

            pareto_front.append((partition_indices, entropy_score, semantic_score))
            scores.append((entropy_score, semantic_score))

    # Convert indices to text for all Pareto-optimal solutions
    final_results = []
    for partition_indices, entropy_score, semantic_score in pareto_front:
        partition_text = [[s for s in subsequences[idx]] for idx in partition_indices]
        partition_probs = [
            [round(float(y), 3) for y in predictions[idx]] for idx in partition_indices
        ]

        final_results.append(
            {
                "text": partition_text,
                "probs": partition_probs,
                "entropy_score": round(float(entropy_score), 3),
                "semantic_score": round(float(semantic_score), 3),
            }
        )

    return final_results


def process_tsv_file(input_file_path, output_file_path):
    """Process texts from TSV file and save Pareto-optimal segmentations."""
    df = pd.read_csv(input_file_path, sep="\t", header=None)

    for idx, text in enumerate(df[1]):
        print(f"\nProcessing text {idx + 1}/{len(df)}")

        # First split into sentences
        sentences = split_into_sentences(text)

        # Combine short sentences if needed
        sentences = combine_short_sentences(sentences)

        # Get Pareto-optimal segmentations
        pareto_solutions = generate_partitionings_with_pareto(sentences)

        # Save results
        result = {"text_id": idx, "pareto_solutions": pareto_solutions}

        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
        for i, sol in enumerate(pareto_solutions):
            print(f"\nSolution {i+1}:")
            print(f"Entropy score: {sol['entropy_score']}")
            print(f"Semantic score: {sol['semantic_score']}")
            for part, probs in zip(sol["text"], sol["probs"]):
                print(f"\nSegment: {' '.join(part)}")
                print(f"Probabilities: {probs}")


# Example usage
if __name__ == "__main__":
    # Get file name from sys argv
    input_file = sys.argv[1]

    # Output file, add _results before extension
    output_file = input_file.replace(".tsv", "_results.jsonl")

    process_tsv_file(input_file, output_file)
