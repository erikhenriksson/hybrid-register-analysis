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


def build_segmentation_trees(sentences):
    """Build all possible valid segmentation trees."""
    text = " ".join(sentences)
    probs, embeddings = predict_and_embed_batch([text])
    registers = get_strong_registers(probs[0])

    if len(text) < min_chars_per_segment:
        return [
            {"segments": [sentences], "probs": [probs[0]], "embeddings": embeddings}
        ]

    possible_trees = []

    # Try all possible split points
    for i in range(1, len(sentences)):
        left_sentences = sentences[:i]
        right_sentences = sentences[i:]
        left_text = " ".join(left_sentences)
        right_text = " ".join(right_sentences)

        # Check minimum length
        if (
            len(left_text) < min_chars_per_segment
            or len(right_text) < min_chars_per_segment
        ):
            continue

        split_texts = [left_text, right_text]
        split_probs, split_embeddings = predict_and_embed_batch(split_texts)
        left_registers = set(get_strong_registers(split_probs[0]))
        right_registers = set(get_strong_registers(split_probs[1]))

        # Skip if either segment has no strong registers or same registers
        if (
            not left_registers
            or not right_registers
            or left_registers == right_registers
        ):
            continue

        # Recursively build subtrees
        left_trees = build_segmentation_trees(left_sentences)
        right_trees = build_segmentation_trees(right_sentences)

        # Combine all possible left and right subtrees
        for left_tree in left_trees:
            for right_tree in right_trees:
                tree = {
                    "segments": left_tree["segments"] + right_tree["segments"],
                    "probs": left_tree["probs"] + right_tree["probs"],
                    "embeddings": torch.cat(
                        [left_tree["embeddings"], right_tree["embeddings"]], dim=0
                    ),
                    "children": [left_tree, right_tree],
                }
                possible_trees.append(tree)

    # Include the unsplit version
    possible_trees.append(
        {"segments": [sentences], "probs": [probs[0]], "embeddings": embeddings}
    )

    return possible_trees


def verify_label_constraints(tree, required_labels):
    """Check if a tree satisfies our label constraints."""
    found_labels = set()

    def collect_labels(subtree):
        for probs in subtree["probs"]:
            found_labels.update(get_strong_registers(probs))
        if "children" in subtree:
            for child in subtree["children"]:
                collect_labels(child)

    collect_labels(tree)
    return required_labels.issubset(found_labels)


def find_best_valid_tree(trees, required_labels):
    """Find the best tree that satisfies our constraints."""
    valid_trees = [
        tree for tree in trees if verify_label_constraints(tree, required_labels)
    ]
    if not valid_trees:
        return None

    def score_tree(tree):
        if "children" not in tree:
            return 0
        left, right = tree["children"]
        left_regs = set(get_strong_registers(left["probs"][0]))
        right_regs = set(get_strong_registers(right["probs"][0]))
        score = len(left_regs ^ right_regs)
        return score + score_tree(left) + score_tree(right)

    return max(valid_trees, key=score_tree)


def recursive_segment(sentences):
    # Get document-level labels
    text = " ".join(sentences)
    doc_probs, doc_embeddings = predict_and_embed_batch([text])
    required_labels = set(get_strong_registers(doc_probs[0]))

    # Build all possible trees
    possible_trees = build_segmentation_trees(sentences)

    # Find best valid tree
    best_tree = find_best_valid_tree(possible_trees, required_labels)

    if best_tree is None:
        return [sentences], [doc_probs[0]], doc_embeddings

    return best_tree["segments"], best_tree["probs"], best_tree["embeddings"]


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

        # Recursively split text
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
