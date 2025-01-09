import json
import sys

import pandas as pd
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Parameters
max_tokens = 512
batch_size = 1
min_chars_per_segment = 250
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
    # Iterate the label list and grab if indices has it
    return [labels_list[i] for i in range(len(labels_list)) if i in indices]


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

    # binary_registers = zero_parents(binary_registers)
    indices = [i for i, p in enumerate(binary_registers) if p]
    return set(indices)


def recursive_segment(sentences, depth=0):
    """
    Returns: (segments, probs, embeddings)
    """
    text = " ".join(sentences)
    print(
        f"Depth {depth}: Processing segment of length {len(text)} with {len(sentences)} sentences"
    )

    # Get current segment's registers - these must be preserved in any split
    base_probs, base_embeddings = predict_and_embed_batch([text])
    base_registers = set(get_strong_registers(base_probs[0]))
    print(f"Depth {depth}: Found registers: {base_registers}")

    if len(text) < min_chars_per_segment:
        print(f"Depth {depth}: Segment too small, returning")
        return [sentences], [base_probs[0]], base_embeddings

    # Try every possible split point
    best_segmentation = None
    best_score = -1

    for i in range(1, len(sentences)):
        print(f"Depth {depth}: Trying split at position {i}/{len(sentences)}")
        left_sentences = sentences[:i]
        right_sentences = sentences[i:]

        # Check minimum lengths
        left_text = " ".join(left_sentences)
        right_text = " ".join(right_sentences)
        if (
            len(left_text) < min_chars_per_segment
            or len(right_text) < min_chars_per_segment
        ):
            continue

        # Get predictions for this split
        split_texts = [left_text, right_text]
        probs, embeddings = predict_and_embed_batch(split_texts)
        left_registers = set(get_strong_registers(probs[0]))
        right_registers = set(get_strong_registers(probs[1]))

        print(f"Depth {depth}: Left registers: {left_registers}")
        print(f"Depth {depth}: Right registers: {right_registers}")

        # Must find all base registers in the union of left and right
        if not base_registers.issubset(left_registers.union(right_registers)):
            continue

        split_score = len(left_registers ^ right_registers)
        if split_score == 0:  # Skip if registers are identical
            continue

        # Recursively split both sides
        left_segments, left_probs, left_embs = recursive_segment(
            left_sentences, depth + 1
        )
        right_segments, right_probs, right_embs = recursive_segment(
            right_sentences, depth + 1
        )

        # Store this as best split if it has highest register difference
        if split_score > best_score:
            best_score = split_score
            best_segmentation = (
                left_segments + right_segments,
                left_probs + right_probs,
                torch.cat([left_embs, right_embs], dim=0),
            )

    # If no valid split found, return current segment
    if best_segmentation is None:
        print(f"Depth {depth}: No valid split found, returning unsplit")
        return [sentences], [base_probs[0]], base_embeddings

    print(f"Depth {depth}: Returning best split with score {best_score}")
    return best_segmentation


def process_tsv_file(input_file_path, output_file_path):
    """Process texts from TSV file using recursive splitting approach."""

    df = pd.read_csv(input_file_path, sep="\t", header=None)

    for idx, row in df.iterrows():
        true_labels = row[0]  # Get true labels from first column
        text = row[1]  # Get text from second column

        # Preprocess text
        truncated_text = truncate_text_to_tokens(text)
        sentences = split_into_sentences(truncated_text)

        # Get document level predictions first
        full_text = " ".join(sentences)
        doc_probs, doc_embeddings = predict_and_embed_batch([full_text])
        document_labels = index_to_name(get_strong_registers(doc_probs[0]))

        # Recursively split text - now handling 5 return values
        segments, segment_probs, segment_embeddings = recursive_segment(sentences)

        # Create result dictionary
        result = {
            "document_labels": document_labels,
            "true_labels": true_labels,
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
        print(f"True labels: {true_labels}")
        print(f"Document-level predicted labels: {document_labels}")
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
