import json
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def calculate_prediction_similarity(
    pred1: List[float], pred2: List[float], threshold: float = 0.3
) -> bool:
    """
    Calculate similarity between two register predictions.
    Returns True if predictions are similar, False if significantly different.
    """
    differences = [abs(p1 - p2) for p1, p2 in zip(pred1, pred2)]
    return max(differences) < threshold


def get_window_prediction(
    sentences: List[str], start_idx: int, window_size: int, model, tokenizer, device
) -> List[float]:
    """
    Get prediction for a specific window of sentences.
    """
    # Handle cases where window might extend beyond text boundaries
    end_idx = min(start_idx + window_size, len(sentences))
    window_text = " ".join(sentences[start_idx:end_idx])

    # Tokenize and prepare input
    inputs = tokenizer(
        window_text, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]

    return [
        round(float(p), 3) for p in probs[:9]
    ]  # First 9 probabilities for registers


def sliding_window_segmentation(
    sentences: List[str], model, tokenizer, device, window_size: int = 3
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Segment text using sliding window approach with precise change point detection.

    Args:
        sentences: List of sentences to segment
        model: The loaded model for predictions
        tokenizer: The tokenizer for the model
        device: Device to run model on (cuda/cpu)
        window_size: Size of sliding window

    Returns:
        Tuple containing:
        - List of segments (each segment is a list of sentences)
        - List of predictions for each segment
    """
    if len(sentences) < window_size:
        text = " ".join(sentences)
        pred = get_window_prediction(
            sentences, 0, len(sentences), model, tokenizer, device
        )
        return [sentences], [pred]

    # Get predictions for all windows
    windows_predictions = []
    for i in range(len(sentences) - window_size + 1):
        pred = get_window_prediction(
            sentences, i, window_size, model, tokenizer, device
        )
        windows_predictions.append(pred)

    # Find rough segment boundaries based on prediction changes
    rough_boundaries = [0]  # Start with first sentence
    for i in range(len(windows_predictions) - 1):
        if not calculate_prediction_similarity(
            windows_predictions[i], windows_predictions[i + 1]
        ):
            rough_boundaries.append(i + window_size // 2)
    rough_boundaries.append(len(sentences))  # Add end boundary

    # Refine boundary positions using precise change point detection
    refined_boundaries = [0]  # Always start with first sentence

    for boundary in rough_boundaries[1:-1]:  # Skip first and last boundaries
        # Look at 2 sentences before and after the rough boundary
        start_idx = max(0, boundary - 2)
        end_idx = min(len(sentences), boundary + 3)

        # Generate all possible split points
        max_difference = 0
        best_split = boundary

        for split in range(start_idx + 1, end_idx):
            # Create before and after windows
            before_pred = get_window_prediction(
                sentences, max(0, split - 2), 2, model, tokenizer, device
            )
            after_pred = get_window_prediction(
                sentences, split, 2, model, tokenizer, device
            )

            # Calculate difference between predictions
            diff = sum(abs(b - a) for b, a in zip(before_pred, after_pred))

            if diff > max_difference:
                max_difference = diff
                best_split = split

        refined_boundaries.append(best_split)

    refined_boundaries.append(len(sentences))  # Add end boundary

    # Create segments and get their predictions
    segments = []
    segment_predictions = []

    for i in range(len(refined_boundaries) - 1):
        start = refined_boundaries[i]
        end = refined_boundaries[i + 1]
        segment = sentences[start:end]
        segments.append(segment)

        # Get prediction for entire segment
        segment_pred = get_window_prediction(
            sentences, start, end - start, model, tokenizer, device
        )
        segment_predictions.append(segment_pred)

    return segments, segment_predictions


def process_text_file(
    input_file: str,
    output_file: str,
    model_name: str = "TurkuNLP/web-register-classification-multilingual",
):
    """
    Process texts from a file using sliding window segmentation.

    Args:
        input_file: Path to input TSV file
        output_file: Path to output JSONL file
        model_name: Name of the model to use for register classification
    """
    # Setup model
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Read input file
    df = pd.read_csv(input_file, sep="\t", header=None)

    # Process each text
    for idx, text in enumerate(df[1]):
        # Tokenize text into sentences (assuming you have the split_into_sentences function)
        sentences = split_into_sentences(text)

        # Perform segmentation
        segments, predictions = sliding_window_segmentation(
            sentences, model, tokenizer, device
        )

        # Create result dictionary
        result = {
            "segments": [" ".join(segment) for segment in segments],
            "predictions": predictions,
            "segment_boundaries": [len(segment) for segment in segments],
        }

        # Write to output file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Print progress
        print(f"\nProcessed text {idx + 1}/{len(df)}")
        print("Segments found:", len(segments))
        for segment, preds in zip(segments, predictions):
            print(f"\nSegment text: {' '.join(segment)}")
            print(f"Predictions: {preds}")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py input_file.tsv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace(".tsv", "_segmented.jsonl")

    process_text_file(input_file, output_file)
