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
model = model.to(device)
model.eval()

# Load spaCy
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Register labels
labels_all = ["MT", "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP"]


def combine_short_sentences(sentences, min_words=5):

    def count_words(sentence):
        return len(sentence.split())

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

    return result


def predict_and_embed_batch(texts, batch_size=32):
    """Predict probabilities and get embeddings for a batch of texts."""
    all_probs = []
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            batch_probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            last_hidden_state = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_state[:, 0, :].cpu()

        batch_probs = [[round(prob, 3) for prob in probs[:9]] for probs in batch_probs]
        all_probs.extend(batch_probs)
        all_embeddings.append(cls_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_probs, all_embeddings


def split_into_sentences(text):
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def truncate_text_to_tokens(text):
    """Truncate text to fit within model's token limit."""
    tokens = tokenizer(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)


def score_split(left_sentences, right_sentences):
    """
    Check if register predictions differ between two segments.
    Returns whether registers differ and predictions for both parts.
    """
    # Get predictions for parts
    left_text = " ".join(left_sentences)
    right_text = " ".join(right_sentences)

    left_pred, _ = predict_and_embed_batch([left_text], batch_size=1)
    right_pred, _ = predict_and_embed_batch([right_text], batch_size=1)

    # Convert to binary predictions using 0.4 threshold
    left_binary = [1 if p >= 0.4 else 0 for p in left_pred[0]]
    right_binary = [1 if p >= 0.4 else 0 for p in right_pred[0]]

    # Check if any register differs
    differs = any(l != r for l, r in zip(left_binary, right_binary))

    return differs, left_pred[0], right_pred[0]


def recursive_split(sentences, min_sentences=5):
    """
    Recursively split text when register predictions differ significantly.
    Returns list of segments and their predictions.
    """
    if len(sentences) < min_sentences * 2:
        text = " ".join(sentences)
        pred, _ = predict_and_embed_batch([text], batch_size=1)
        return [(sentences, pred[0])]

    # Try all possible splits
    best_score = -float("inf")
    best_split = None
    best_preds = None

    for i in range(min_sentences, len(sentences) - min_sentences + 1):
        left = sentences[:i]
        right = sentences[i:]
        score, left_pred, right_pred = score_split(left, right)

        if score > best_score:
            best_score = score
            best_split = i
            best_preds = (left_pred, right_pred)

    # Only split if registers differ
    if not best_score:  # If no registers differ
        text = " ".join(sentences)
        pred, _ = predict_and_embed_batch([text], batch_size=1)
        return [(sentences, pred[0])]

    # Recurse on both parts
    left_segments = recursive_split(sentences[:best_split])
    right_segments = recursive_split(sentences[best_split:])

    return left_segments + right_segments


def get_dominant_registers(probs, threshold=0.4):
    """Get names of registers that pass the threshold."""
    dominant = [labels_all[i] for i, p in enumerate(probs) if p >= threshold]
    return dominant if dominant else ["None"]


def process_tsv_file(input_file_path, output_file_path):
    """Process texts from TSV file using recursive splitting approach."""
    df = pd.read_csv(input_file_path, sep="\t", header=None)

    for idx, text in enumerate(df[1]):
        # Preprocess text
        truncated_text = truncate_text_to_tokens(text)
        sentences = split_into_sentences(truncated_text)

        sentences = combine_short_sentences(sentences)

        # Get segments using recursive splitting
        segments_with_preds = recursive_split(sentences)
        segments = [seg for seg, _ in segments_with_preds]
        segment_probs = [pred for _, pred in segments_with_preds]

        # Get embeddings for final segments
        segment_texts = [" ".join(segment) for segment in segments]
        _, segment_embeddings = predict_and_embed_batch(
            segment_texts, batch_size=batch_size
        )

        # Create result dictionary
        result = {
            "segments": segments,
            "segment_probs": [
                [float(prob) for prob in probs] for probs in segment_probs
            ],
            "segment_embeddings": segment_embeddings.tolist(),
        }

        # Write to JSONL file
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Print progress and results
        print(f"\nProcessed text {idx + 1}/{len(df)}")
        print("Predictions for each segment:")
        for segment, probs in zip(segments, segment_probs):
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
    output_file = input_file.replace(".tsv", "_results_recursive.jsonl")
    process_tsv_file(input_file, output_file)
