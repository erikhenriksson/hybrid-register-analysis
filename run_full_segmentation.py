from itertools import combinations
import sys, json

import pandas as pd
import spacy
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# At the start of your script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
model_name = "TurkuNLP/web-register-classification-multilingual"
tokenizer_name = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model = model.to(device)
model.eval()
# Get label mapping - only first 9 labels
label_names = {i: model.config.id2label[i] for i in range(9)}
label_to_id = {v: k for k, v in label_names.items()}

# Load spaCy
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Default settings
MIN_WORDS = 15
MAX_TOKENS = 512

prediction_cache = {}


def get_model_predictions(text):
    if text in prediction_cache:
        return prediction_cache[text]

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_TOKENS
    )
    # Move input tensors to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # Move predictions back to CPU before converting to numpy
    probs = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()

    prediction_cache[text] = probs
    return probs


def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def truncate_text_to_tokens(text):
    """Truncate text to fit within model's token limit"""
    tokens = tokenizer(text, truncation=True, max_length=MAX_TOKENS)
    truncated_text = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
    return truncated_text


def combine_short_sentences(sentences, min_words=MIN_WORDS, max_segments=20):
    """
    Combine sentences adaptively by increasing min_words until we get fewer than max_segments.
    """

    def _combine_with_min_words(sentences, min_words):
        if not sentences:
            return []

        combined_sentences = []
        current_segment = []
        current_count = 0

        for sentence in sentences:
            word_count = len(word_tokenize(sentence))

            if current_count + word_count < min_words:
                current_segment.append(sentence)
                current_count += word_count
            else:
                if current_segment:
                    combined_sentences.append(" ".join(current_segment))
                current_segment = [sentence]
                current_count = word_count

        if current_segment:
            combined_sentences.append(" ".join(current_segment))

        return combined_sentences

    # Start with initial min_words
    current_min_words = min_words
    while True:
        combined = _combine_with_min_words(sentences, current_min_words)
        if len(combined) <= max_segments:
            print(
                f"Combined to {len(combined)} segments with min_words={current_min_words}"
            )
            return combined

        # Increase min_words by 1 and try again
        current_min_words += 1
        print(
            f"Too many segments ({len(combined)}), increasing min_words to {current_min_words}"
        )


def generate_ordered_partitions(sentences):
    """Generate all possible ordered partitions of sentences."""
    n = len(sentences)

    def partition_from_indices(indices):
        """Convert partition indices to actual sentence groups."""
        result = []
        prev = 0
        for index in indices:
            result.append(" ".join(sentences[prev:index]))
            prev = index
        result.append(" ".join(sentences[prev:]))
        return result

    all_partitions = []
    # Generate all possible combinations of partition points
    for r in range(n):  # r is the number of partition points
        for indices in combinations(
            range(1, n), r
        ):  # Partition points are between 1 and n-1
            all_partitions.append(partition_from_indices(indices))

    return all_partitions


def get_unique_segments_from_partitions(partitions):
    """Extract all unique segments that appear in any partition."""
    unique_segments = set()
    for partition in partitions:
        unique_segments.update(partition)
    return list(unique_segments)


def analyze_hybrid_discreteness(text):
    prediction_cache.clear()

    # First truncate the text
    truncated_text = truncate_text_to_tokens(text)
    probs = get_model_predictions(truncated_text)

    initial_sentences = split_into_sentences(truncated_text)
    sentences = combine_short_sentences(initial_sentences)
    print(f"\nSentence processing:")
    print(f"Initial sentences: {len(initial_sentences)}")
    print(f"After combining short sentences: {len(sentences)}")

    partitions = generate_ordered_partitions(sentences)
    print(f"Generated {len(partitions)} possible partitions")

    # Get all unique segments and their predictions upfront
    unique_segments = get_unique_segments_from_partitions(partitions)
    print(f"Found {len(unique_segments)} unique segments")

    # Pre-compute predictions for all unique segments
    segment_predictions_map = {}
    for segment in tqdm(
        unique_segments, desc="Computing predictions for unique segments"
    ):
        segment_predictions_map[segment] = get_model_predictions(segment)

    # Calculate entropy for each partition using pre-computed predictions
    best_entropy = float("inf")
    best_partition = None
    best_predictions = None

    for partition in tqdm(partitions, desc="Evaluating partitions"):
        # Get predictions for each segment from our pre-computed map
        segment_predictions = [
            segment_predictions_map[segment] for segment in partition
        ]

        # Calculate average entropy across segments
        total_entropy = 0
        for pred in segment_predictions:
            epsilon = 1e-10
            entropy = -np.sum(pred * np.log2(pred + epsilon))
            total_entropy += entropy
        avg_entropy = total_entropy / len(segment_predictions)

        if avg_entropy < best_entropy:
            best_entropy = avg_entropy
            best_partition = partition
            best_predictions = segment_predictions

    # Return with all necessary conversions done here
    return {
        "full_text_prediction": [round(x, 4) for x in probs.tolist()],
        "segments": best_partition,
        "segment_predictions": [
            [round(x, 4) for x in pred.tolist()] for pred in best_predictions
        ],
        "segment_lengths": [len(word_tokenize(seg)) for seg in best_partition],
        "n_segments": len(best_partition),
        "best_entropy": round(float(best_entropy), 4),
    }


def process_file(file_path):
    output_file = file_path + "_results.jsonl"
    print(f"Will write results to: {output_file}")

    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        print(f"Found {total_lines} lines in input file")
        f.seek(0)

        for idx, line in tqdm(enumerate(f), total=total_lines):
            text = line.strip()
            result = line[line.find("\t") + 1 :]
            if not text:
                print(f"Skipping empty line at index {idx}")
                continue

            try:
                print(f"\nProcessing document {idx} with length {len(text)}")
                result = analyze_hybrid_discreteness(text)
                result["document_id"] = idx

                # Simply write the result
                with open(output_file, "a", encoding="utf-8") as f_out:
                    json_string = json.dumps(result, ensure_ascii=False)
                    f_out.write(json_string + "\n")
                    print(f"Successfully wrote result for document {idx}")

            except Exception as e:
                print(f"Error processing document {idx}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue


process_file(sys.argv[1])
