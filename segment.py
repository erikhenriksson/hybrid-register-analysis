import json
import sys

import pandas as pd
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Parameters
max_tokens = 512
batch_size = 1
min_chars_per_segment = 500
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


# Get indices of registers that pass the threshold
def get_strong_registers(probs):
    binary_registers = [int(p >= threshold) for p in probs]

    # Zero out parents
    
    #binary_registers = zero_parents(binary_registers)
    indices = [i for i, p in enumerate(binary_registers) if p]
    return set(indices)


def recursive_segment(sentences, parent_registers=None, required_labels=None):
    text = " ".join(sentences)
    if parent_registers is None:
        probs, embeddings = predict_and_embed_batch([text])
        parent_registers = get_strong_registers(probs[0])
        required_labels = set(parent_registers)  # These are the labels we must find

        if len(text) < min_chars_per_segment or not parent_registers:
            return [sentences], probs, embeddings

    best_split = None
    best_split_score = -1
    best_probs_left = None
    best_probs_right = None
    best_emb_left = None
    best_emb_right = None
    best_labels_covered = None

    # Try all possible split points
    for i in range(1, len(sentences)):
        left_text = " ".join(sentences[:i])
        right_text = " ".join(sentences[i:])

        if (
            len(left_text) < min_chars_per_segment
            or len(right_text) < min_chars_per_segment
        ):
            continue

        split_texts = [left_text, right_text]
        probs, embeddings = predict_and_embed_batch(split_texts)
        left_registers = set(get_strong_registers(probs[0]))
        right_registers = set(get_strong_registers(probs[1]))

        # Skip if either segment has no strong registers
        if not left_registers or not right_registers:
            continue

        # Calculate which required labels are covered by this split
        labels_covered = left_registers.union(right_registers)

        # Skip if we don't cover all required labels
        if not required_labels.issubset(labels_covered):
            continue

        # Calculate split score
        split_score = len(left_registers ^ right_registers)

        if split_score > best_split_score:
            best_split_score = split_score
            best_split = i
            best_probs_left = probs[0]
            best_probs_right = probs[1]
            best_emb_left = embeddings[0].unsqueeze(0)
            best_emb_right = embeddings[1].unsqueeze(0)
            best_labels_covered = labels_covered

    if best_split is None:
        probs, embeddings = predict_and_embed_batch([text])
        return [sentences], probs, embeddings

    left_sentences = sentences[:best_split]
    right_sentences = sentences[best_split:]

    left_registers = set(get_strong_registers(best_probs_left))
    right_registers = set(get_strong_registers(best_probs_right))

    # When making recursive calls, pass down the labels that each branch must cover
    left_required = required_labels.intersection(left_registers)
    right_required = required_labels.intersection(right_registers)

    left_segments, left_probs, left_embeddings = recursive_segment(
        left_sentences, left_registers, left_required
    )
    right_segments, right_probs, right_embeddings = recursive_segment(
        right_sentences, right_registers, right_required
    )

    if len(left_segments) == 1 and len(right_segments) == 1:
        segments = [left_sentences, right_sentences]
        segment_probs = [[p for p in best_probs_left], [p for p in best_probs_right]]
        segment_embeddings = torch.cat([best_emb_left, best_emb_right], dim=0)
    else:
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
