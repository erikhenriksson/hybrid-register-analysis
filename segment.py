import json
import sys

import pandas as pd
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Parameters
max_tokens = 512
batch_size = 1
min_words_per_segment = 50
threshold = 0.4

# Load register classification model
model_name = "TurkuNLP/web-register-classification-multilingual"
tokenizer_name = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load spaCy
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Labels
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

# Flat list of labels
labels_list = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]

# Sublabel index to parent index
index_to_parent = {
    labels_list.index(sublabel): labels_list.index(parent)
    for parent, sublabels in labels_structure.items()
    for sublabel in sublabels
}


# Zero parents when child active
def zero_parents(binary_list):
    for child_idx, parent_idx in index_to_parent.items():
        if child_idx < len(binary_list) and binary_list[child_idx] == 1:
            binary_list[parent_idx] = 0
    return binary_list


# Index to name mapping
def index_to_name(indices):
    return [labels_list[i] for i in indices]


# Predict probabilities and get embeddings for a batch of texts
def predict_and_embed_batch(texts, batch_size=batch_size):
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

        all_probs.extend(batch_probs)
        all_embeddings.append(cls_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_probs, all_embeddings


# Split text into sentences
def split_into_sentences(text):
    """Split text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


# Truncate text to fit within model's token limit
def truncate_text_to_tokens(text):
    """Truncate text to fit within model's token limit."""
    tokens = tokenizer(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)


# Count words in a text segment
def get_word_count(text):
    """Count words in a text segment."""
    return len(text.split())


# Get indices of registers that pass the threshold (without parents)
def get_strong_registers(probs):
    binary_registers = [int(p >= threshold) for p in probs]
    parents_zeroed = zero_parents(binary_registers)
    indices = [i for i, p in enumerate(parents_zeroed) if p]
    return indices


# Recursively segment text based on register predictions.
def recursive_segment(sentences, parent_registers=None):
    # If we don't have parent registers (first call), analyze full text
    text = " ".join(sentences)
    if parent_registers is None:
        probs, embeddings = predict_and_embed_batch([text])
        parent_registers = get_strong_registers(probs[0])

        # If text is too short or no strong registers, return as is
        if get_word_count(text) < min_words_per_segment or not parent_registers:
            return [sentences], probs, embeddings

    best_split = None
    best_split_score = -1
    best_probs_left = None
    best_probs_right = None
    best_emb_left = None
    best_emb_right = None

    # Try all possible split points
    for i in range(1, len(sentences)):
        left_text = " ".join(sentences[:i])
        right_text = " ".join(sentences[i:])

        # Check minimum length requirement
        if (
            get_word_count(left_text) < min_words_per_segment
            or get_word_count(right_text) < min_words_per_segment
        ):
            continue

        # Get predictions for both segments
        split_texts = [left_text, right_text]
        probs, embeddings = predict_and_embed_batch(split_texts)
        left_registers = get_strong_registers(probs[0])
        right_registers = get_strong_registers(probs[1])

        # Skip if either segment has no strong registers
        if not left_registers or not right_registers:
            continue

        # Check if at least one segment maintains continuity with parent
        if not (
            left_registers & parent_registers or right_registers & parent_registers
        ):
            continue

        # Check if segments have different register patterns
        if left_registers == right_registers:
            continue

        # Calculate split score (number of different strong registers)
        split_score = len(left_registers ^ right_registers)

        if split_score > best_split_score:
            best_split_score = split_score
            best_split = i
            best_probs_left = probs[0]
            best_probs_right = probs[1]
            best_emb_left = embeddings[0].unsqueeze(0)
            best_emb_right = embeddings[1].unsqueeze(0)

    # If no valid split found, return current segment with its predictions
    if best_split is None:
        probs, embeddings = predict_and_embed_batch([text])
        return [sentences], probs, embeddings

    # Recursively process both segments
    left_sentences = sentences[:best_split]
    right_sentences = sentences[best_split:]

    # Get registers for recursive calls
    left_registers = get_strong_registers(best_probs_left)
    right_registers = get_strong_registers(best_probs_right)

    # Make recursive calls with corresponding registers
    left_segments, left_probs, left_embeddings = recursive_segment(
        left_sentences, left_registers
    )
    right_segments, right_probs, right_embeddings = recursive_segment(
        right_sentences, right_registers
    )

    # If no further splits occurred, use the predictions from this level
    if len(left_segments) == 1 and len(right_segments) == 1:
        segments = [left_sentences, right_sentences]
        segment_probs = [[p for p in best_probs_left], [p for p in best_probs_right]]
        segment_embeddings = torch.cat([best_emb_left, best_emb_right], dim=0)
    else:
        # Further splits occurred, concatenate all results in order
        segments = left_segments + right_segments
        segment_probs = left_probs + right_probs
        segment_embeddings = torch.cat([left_embeddings, right_embeddings], dim=0)

    return segments, segment_probs, segment_embeddings


def process_tsv_file(input_file_path, output_file_path):
    """Process texts from TSV file using recursive splitting approach."""

    df = pd.read_csv(input_file_path, sep="\t", header=None)

    for idx, text in enumerate(df[1]):
        # Preprocess text
        truncated_text = truncate_text_to_tokens(text)
        sentences = split_into_sentences(truncated_text)

        # Get document level predictions first
        full_text = " ".join(sentences)
        doc_probs, doc_embeddings = predict_and_embed_batch([full_text])
        document_labels = index_to_name(get_strong_registers(doc_probs[0]))

        # Recursively split text
        segments, segment_probs, segment_embeddings = recursive_segment(sentences)

        # Create result dictionary
        result = {
            "document_labels": document_labels,
            "document_embeddings": doc_embeddings.tolist(),
            "segments": segments,
            "segment_labels": [
                index_to_name(get_strong_registers(probs)) for probs in segment_probs
            ],
            "segment_probs": [
                [round(float(prob), 3) for prob in probs] for probs in segment_probs
            ],
            "segment_embeddings": segment_embeddings.tolist(),
        }

        # Write to JSONL file
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Print progress and results
        print(f"\nProcessed text {idx + 1}/{len(df)}")
        print(f"Document-level labels: {document_labels}")
        print("\nPredictions for each segment:")

        for segment, probs, segment_labels in zip(
            result["segments"], result["segment_probs"], result["segment_labels"]
        ):
            print(f"Segment labels: {segment_labels}")
            print(f"Text: {' '.join(segment)}")
            print(f"Probabilities: {probs}\n")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 segment.py input_file.tsv output_file.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_tsv_file(input_file, output_file)
