import pandas as pd
import numpy as np
import torch
import spacy
import json
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# [Previous model loading and helper functions remain the same until process_text]


def process_text_recursive(text: str) -> Dict:
    """Process a text recursively, attempting to split it and its resulting segments."""
    # Split into sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # If text is too short, return it as a leaf segment
    total_text = " ".join(sentences)
    if len(total_text) < 500:  # Too short to split into two 250-char segments
        return {
            "text": total_text,
            "sentences": sentences,
            "predictions": get_predictions(total_text),
            "is_leaf": True,
        }

    # If we don't have enough sentences, return as leaf
    if len(sentences) < 2:
        return {
            "text": total_text,
            "sentences": sentences,
            "predictions": get_predictions(total_text),
            "is_leaf": True,
        }

    # Get embeddings for sentences
    embeddings = get_embeddings(sentences)

    # Generate binary splits that meet minimum length requirement
    splits = []
    for i in range(1, len(sentences)):
        left_indices = list(range(i))
        right_indices = list(range(i, len(sentences)))

        # Get text for each potential segment
        left_text = " ".join([sentences[j] for j in left_indices])
        right_text = " ".join([sentences[j] for j in right_indices])

        # Only include split if both segments meet minimum length
        if len(left_text) >= 250 and len(right_text) >= 250:
            splits.append((left_indices, right_indices))

    # If no valid splits are found, return as leaf
    if not splits:
        return {
            "text": total_text,
            "sentences": sentences,
            "predictions": get_predictions(total_text),
            "is_leaf": True,
        }

    # Find optimal split
    best_split = find_optimal_split(embeddings, splits)
    split_indices = best_split["split"]

    # Get text for each segment
    segment1_text = " ".join([sentences[i] for i in split_indices[0]])
    segment2_text = " ".join([sentences[i] for i in split_indices[1]])

    # Recursively process each segment
    segment1_analysis = process_text_recursive(segment1_text)
    segment2_analysis = process_text_recursive(segment2_text)

    return {
        "text": total_text,
        "sentences": sentences,
        "predictions": get_predictions(total_text),
        "is_leaf": False,
        "split_metrics": best_split["metrics"],
        "segments": [segment1_analysis, segment2_analysis],
    }


def process_tsv_file(input_file_path: str, output_file_path: str):
    """Process texts from TSV file and save recursive segmentations with predictions."""
    df = pd.read_csv(
        input_file_path, sep="\t", header=None, na_values="", keep_default_na=False
    )

    def print_segment_tree(segment: Dict, depth: int = 0):
        """Helper function to print the segment tree structure."""
        indent = "  " * depth
        print(f"{indent}Text length: {len(segment['text'])} chars")
        print(f"{indent}Predictions: {', '.join(segment['predictions'])}")

        if not segment["is_leaf"]:
            print(f"{indent}Split metrics: {segment['split_metrics']}")
            print(f"{indent}Segment 1:")
            print_segment_tree(segment["segments"][0], depth + 1)
            print(f"{indent}Segment 2:")
            print_segment_tree(segment["segments"][1], depth + 1)
        else:
            print(f"{indent}[Leaf segment]")

    with open(output_file_path, "w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            true_labels = row[0].split()  # Assuming labels are space-separated
            text = row[1]
            text = truncate_text_to_tokens(text)

            # Process the text recursively
            results = process_text_recursive(text)

            print(f"\nDocument {idx} (true labels: {true_labels}):")
            print_segment_tree(results)
            print("\n")

            # Add metadata and write to JSONL
            output_record = {
                "text_id": idx,
                "true_labels": true_labels,
                "analysis": results,
            }
            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 segment.py input_file.tsv output_file.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_tsv_file(input_file, output_file)
