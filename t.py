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
min_words = 15
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

            batch_probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            last_hidden_state = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_state[:, 0, :].cpu()

        batch_probs = [[round(prob, 3) for prob in probs[:9]] for probs in batch_probs]

        all_probs.extend(batch_probs)
        all_embeddings.append(cls_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_probs, all_embeddings


def combine_short_sentences(sentences, min_words=50):
    """Combine short sentences to meet minimum word count requirement."""

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

        return result


def split_into_sentences(text):
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def truncate_text_to_tokens(text):
    """Truncate text to fit within model's token limit."""
    tokens = tokenizer(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)


def find_best_cut_point(sentences, window_start, window_size):
    """Find the best place to cut within the overlapping region of two windows."""
    overlap_start = window_start + 1
    overlap_end = window_start + window_size - 1

    best_score = float("-inf")
    best_cut = overlap_start + (overlap_end - overlap_start) // 2

    for cut in range(overlap_start, overlap_end):
        before_text = " ".join(sentences[cut - 2 : cut])
        after_text = " ".join(sentences[cut : cut + 2])

        before_pred, _ = predict_and_embed_batch([before_text], batch_size=1)
        after_pred, _ = predict_and_embed_batch([after_text], batch_size=1)

        before_registers = {j for j, p in enumerate(before_pred[0]) if p >= 0.4}
        after_registers = {j for j, p in enumerate(after_pred[0]) if p >= 0.4}

        score = len(before_registers.symmetric_difference(after_registers))

        if score > best_score:
            best_score = score
            best_cut = cut

    return best_cut


def segment_text(sentences, window_size=5):
    """Segment text using sliding windows with intelligent cut point selection."""
    if len(sentences) <= window_size:
        return [sentences]

    # Get predictions for overlapping windows
    predictions = []
    for i in range(len(sentences) - window_size + 1):
        window = sentences[i : i + window_size]
        text = " ".join(window)
        pred, _ = predict_and_embed_batch([text], batch_size=1)
        predictions.append(pred[0])

    # Find segment boundaries
    boundaries = [0]

    for i in range(len(predictions) - 1):
        current_pred = predictions[i]
        next_pred = predictions[i + 1]

        current_registers = {j for j, p in enumerate(current_pred) if p >= 0.4}
        next_registers = {j for j, p in enumerate(next_pred) if p >= 0.4}

        if len(current_registers.symmetric_difference(next_registers)) >= 1:
            cut = find_best_cut_point(sentences, i, window_size)
            if cut - boundaries[-1] >= window_size // 2:  # Minimum segment size
                boundaries.append(cut)

    boundaries.append(len(sentences))

    # Create segments
    segments = []
    for i in range(len(boundaries) - 1):
        segment = sentences[boundaries[i] : boundaries[i + 1]]
        segments.append(segment)

    return segments


def get_dominant_registers(probs, threshold=0.4):
    """Get names of registers that pass the threshold."""
    dominant = [labels_all[i] for i, p in enumerate(probs) if p >= threshold]
    return dominant if dominant else ["None"]


def process_tsv_file(input_file_path, output_file_path):
    """Process texts from TSV file using the simplified sliding window approach."""
    df = pd.read_csv(input_file_path, sep="\t", header=None)

    for idx, text in enumerate(df[1]):
        # Preprocess text
        truncated_text = truncate_text_to_tokens(text)
        sentences = split_into_sentences(truncated_text)
        combined_sentences = combine_short_sentences(sentences)

        # Get segments
        segments = segment_text(combined_sentences)

        # Get predictions for final segments
        segment_texts = [" ".join(segment) for segment in segments]
        segment_probs, segment_embeddings = predict_and_embed_batch(
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
    output_file = input_file.replace(".tsv", "_results_sliding.jsonl")
    process_tsv_file(input_file, output_file)
