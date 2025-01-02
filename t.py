import json
import sys

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load register classification model
model_name = "TurkuNLP/web-register-classification-multilingual"
tokenizer_name = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_tokens = 512
batch_size = 64
min_words = 50
max_segments = 20
model = model.to(device)
model.eval()

# Load spaCy
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Register labels
labels_all = ["MT", "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


def predict_and_embed_batch(texts, batch_size=32):
    """Predict probabilities and get embeddings for a batch of texts."""
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
            outputs = model(**inputs, output_hidden_states=True)

            # Get predictions
            batch_probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()

            # Get embeddings
            last_hidden_state = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_state[:, 0, :].cpu()

        # Round probabilities
        batch_probs = [[round(prob, 3) for prob in probs[:9]] for probs in batch_probs]

        all_probs.extend(batch_probs)
        all_embeddings.append(cls_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_probs, all_embeddings


def combine_short_sentences(
    sentences, initial_min_words=min_words, max_segments=max_segments
):
    """Combine short sentences to meet minimum word count requirement."""
    min_words = initial_min_words

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
                if count_words(buffer) >= min_words:
                    result.append(buffer.strip())
                    buffer = ""

        if buffer:
            result.append(buffer.strip())

        # Final pass to ensure no short sentences
        i = 0
        while i < len(result):
            if count_words(result[i]) < min_words:
                if i < len(result) - 1:
                    result[i + 1] = result[i] + " " + result[i + 1]
                    result.pop(i)
                elif i > 0:
                    result[i - 1] += " " + result[i]
                    result.pop(i)
                else:
                    break
            else:
                i += 1

        if len(result) <= max_segments:
            return result
        min_words += 1


def split_into_sentences(text):
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def truncate_text_to_tokens(text):
    """Truncate text to fit within model's token limit."""
    tokens = tokenizer(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)


def calculate_prediction_similarity(pred1, pred2, threshold=0.3):
    """
    Calculate similarity between two register predictions.
    Returns True if predictions are similar (i.e., represent same register type)
    """
    # Get dominant registers for each prediction (lowered threshold)
    dom1 = [i for i, p in enumerate(pred1) if p >= 0.3]
    dom2 = [i for i, p in enumerate(pred2) if p >= 0.3]

    # Calculate Jaccard similarity between dominant register sets
    intersection = len(set(dom1).intersection(dom2))
    union = len(set(dom1).union(dom2))
    jaccard = intersection / union if union > 0 else 0

    # If registers are very different (low Jaccard similarity), predictions are different
    if jaccard < 0.3:
        return False

    # Calculate total absolute difference in probabilities
    total_diff = sum(abs(p1 - p2) for p1, p2 in zip(pred1, pred2))

    return total_diff < threshold  # return True if predictions are similar


def sliding_window_segmentation(sentences, window_size=3, min_segment_size=2):
    """
    Segment text using sliding window approach with precise boundary detection.
    Added constraints for minimum segment size and significant register changes.
    """
    if len(sentences) < window_size:
        return [sentences]

    # Get predictions for all possible windows
    windows = []
    predictions = []

    # Use larger windows for initial segmentation
    for i in range(len(sentences) - window_size + 1):
        window = sentences[i : i + window_size]
        text = " ".join(window)
        pred, _ = predict_and_embed_batch([text], batch_size=1)
        windows.append(window)
        predictions.append(pred[0])

    # Find rough segment boundaries based on prediction changes
    segment_boundaries = [0]
    last_boundary = 0

    for i in range(len(predictions) - 1):
        # Only consider boundary if minimum segment size is met
        if (i + window_size // 2 - last_boundary) >= min_segment_size:
            # Check for significant register change
            if not calculate_prediction_similarity(predictions[i], predictions[i + 1]):
                # Verify this isn't a temporary fluctuation by looking ahead
                if i + 2 < len(predictions):
                    # Check if the change persists
                    if not calculate_prediction_similarity(
                        predictions[i], predictions[i + 2]
                    ):
                        segment_boundaries.append(i + window_size // 2)
                        last_boundary = i + window_size // 2

    segment_boundaries.append(len(sentences))

    # Refine boundary positions using precise change point detection
    refined_boundaries = [0]

    for boundary in segment_boundaries[1:-1]:
        start_idx = max(0, boundary - 2)
        end_idx = min(len(sentences), boundary + 3)

        max_difference = 0
        best_split = boundary

        for split in range(start_idx + 1, end_idx):
            # Use larger windows for boundary analysis
            before_text = " ".join(sentences[max(0, split - 3) : split])
            after_text = " ".join(sentences[split : min(len(sentences), split + 3)])

            before_pred, _ = predict_and_embed_batch([before_text], batch_size=1)
            after_pred, _ = predict_and_embed_batch([after_text], batch_size=1)

            # Get dominant registers for both windows
            before_dom = [i for i, p in enumerate(before_pred[0]) if p >= 0.4]
            after_dom = [i for i, p in enumerate(after_pred[0]) if p >= 0.4]

            # Calculate register change score
            register_change = 0 if set(before_dom) == set(after_dom) else 1

            # Calculate probability difference
            prob_diff = sum(abs(b - a) for b, a in zip(before_pred[0], after_pred[0]))

            # Combined score favoring both register changes and probability differences
            total_diff = register_change * 2 + prob_diff

            if total_diff > max_difference:
                max_difference = total_diff
                best_split = split

        # Only keep boundary if there's a significant enough difference
        if max_difference > 0.5:  # Lowered threshold for significance
            refined_boundaries.append(best_split)

    refined_boundaries.append(len(sentences))

    # Create segments based on refined boundaries
    segments = []
    for i in range(len(refined_boundaries) - 1):
        start = refined_boundaries[i]
        end = refined_boundaries[i + 1]
        segment = sentences[start:end]
        segments.append(segment)

    # Validate and merge segments based on full-text register predictions
    final_segments = validate_and_merge_segments(segments)

    return final_segments


def get_dominant_registers(probs, threshold=0.4):
    """Get names of registers that pass the threshold."""
    dominant = [labels_all[i] for i, p in enumerate(probs) if p >= threshold]
    return dominant if dominant else ["None"]


def process_tsv_file_sliding_window(input_file_path, output_file_path):
    """Process texts from TSV file using sliding window approach."""
    df = pd.read_csv(input_file_path, sep="\t", header=None)

    for idx, text in enumerate(df[1]):
        # Preprocess text
        truncated_text = truncate_text_to_tokens(text)
        sentences = split_into_sentences(truncated_text)
        combined_sentences = combine_short_sentences(sentences)

        # Get segments using sliding window
        segments = sliding_window_segmentation(combined_sentences)

        # Get predictions for final segments
        segment_texts = [" ".join(segment) for segment in segments]
        partition_probs, partition_embeddings = predict_and_embed_batch(
            segment_texts, batch_size=batch_size
        )

        # Create result dictionary
        result = {
            "segments": segments,
            "segment_probs": [
                [float(prob) for prob in probs] for probs in partition_probs
            ],
            "segment_embeddings": partition_embeddings.tolist(),
        }

        # Write to JSONL file
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Print progress and results
        print(f"\nProcessed text {idx + 1}/{len(df)}")
        print("Predictions for each segment:")
        for segment, probs in zip(segments, partition_probs):
            dominant_registers = get_dominant_registers(probs)
            print(f"Dominant registers: {', '.join(dominant_registers)}")
            print(f"Text: {' '.join(segment)}")
            print(f"Probabilities: {probs}\n")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py input_file.tsv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".tsv", "_results_sliding.jsonl")
    process_tsv_file_sliding_window(input_file, output_file)
