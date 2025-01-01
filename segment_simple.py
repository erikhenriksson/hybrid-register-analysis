import json
import sys

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BATCH_SIZE = 64
OVERSEGMENTATION_WEIGHT = 1
labels_all = ["MT", "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


def calculate_multilabel_entropy(probabilities):
    probabilities = np.array(probabilities)
    # Calculate binary entropy for each position
    entropy = -(
        probabilities * np.log(probabilities)
        + (1 - probabilities) * np.log(1 - probabilities)
    )

    return np.mean(np.sum(entropy, axis=-1))  # Sum over labels, then mean over batch


def calculate_variance(probabilities):
    """
    Calculate the variance of probabilities within a block.
    """
    return np.var(probabilities, axis=0).mean()


def score_partition(partition_predictions, n):
    num_blocks = len(partition_predictions)

    # Calculate base metrics
    avg_entropy = np.mean(
        [calculate_multilabel_entropy(block) for block in partition_predictions]
    )

    # Calculate variance
    avg_variance = np.mean(
        [calculate_variance(block) for block in partition_predictions]
    )

    # Oversegmentation penalty
    oversegmentation_penalty = num_blocks / n

    # Combine all components
    score = (
        -avg_variance
        - OVERSEGMENTATION_WEIGHT * oversegmentation_penalty  # Add penalties
    )
    return score


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

# Load spaCy
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def predict_and_embed_batch(texts, batch_size=32):
    """
    Predict probabilities and get embeddings for a batch of texts in a single function.

    Args:
        texts (list of str): The input texts.
        batch_size (int): The batch size for processing.

    Returns:
        tuple:
            - all_probs (list of list of float): Predicted probabilities for each text.
            - all_embeddings (torch.Tensor): Embeddings for each text.
    """
    all_probs = []
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        # Get predictions and embeddings
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,  # Request hidden states for embeddings
            )

            # Get predictions (logits converted to probabilities using sigmoid)
            batch_probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()

            # Get embeddings (last hidden state [CLS] token)
            last_hidden_state = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_state[:, 0, :].cpu()

        # Round probabilities to three decimals for consistency
        batch_probs = [[round(prob, 3) for prob in probs[:9]] for probs in batch_probs]

        # Append results
        all_probs.extend(batch_probs)
        all_embeddings.append(cls_embeddings)

    # Concatenate all embeddings into a single tensor
    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_probs, all_embeddings


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
    predictions, embeddings = predict_and_embed_batch(texts, batch_size=BATCH_SIZE)

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

        # Compute the score for this partition
        score = score_partition(partition_predictions, len(sentences))

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
    output_file = input_file.replace(".tsv", "_results_new.jsonl")

    process_tsv_file(input_file, output_file)
