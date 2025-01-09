import pandas as pd
import numpy as np
import torch
import spacy
import json
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# Constants
MIN_CHARS = 300  # Minimum characters for a valid segment

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


def truncate_text_to_tokens(text: str) -> str:
    """Truncate text to fit within model's token limit."""
    tokens = tokenizer(text, truncation=True, max_length=512)
    return tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)


def get_predictions(text: str) -> List[str]:
    """Get multilabel predictions for a text segment."""
    with torch.no_grad():
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        # Apply sigmoid and threshold
        probabilities = torch.sigmoid(outputs.logits)
        predictions = (probabilities > 0.5).squeeze().cpu().numpy()

        # Convert to labels using model's id2label
        predicted_labels = [
            model.config.id2label[i] for i, pred in enumerate(predictions) if pred
        ]

        return predicted_labels


def get_embeddings(sentences: List[str]) -> np.ndarray:
    """Extract embeddings from the model for a list of sentences."""
    embeddings = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)

            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = inputs["attention_mask"]

            token_embeddings = last_hidden_state[0]
            mask = (
                attention_mask[0].unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * mask, 0)
            sum_mask = torch.clamp(mask.sum(0), min=1e-9)
            sentence_embedding = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings.append(sentence_embedding)

    return np.array(embeddings)


def find_register_split(
    splits: List[Tuple[List[int], List[int]]], sentences: List[str]
) -> Dict:
    """Find a single major register change point, if one exists."""

    def get_segment_text(indices: List[int]) -> str:
        return " ".join([sentences[i] for i in indices])

    # Get document-level predictions
    full_text = " ".join(sentences)
    doc_predictions = set(get_predictions(full_text))

    best_split = {"split": None, "metrics": None}

    # For each potential split point
    for split_indices in splits:
        segment1_indices, segment2_indices = split_indices

        # Get predictions for each segment
        text1 = get_segment_text(segment1_indices)
        text2 = get_segment_text(segment2_indices)
        pred1 = set(get_predictions(text1))
        pred2 = set(get_predictions(text2))

        # Only consider it a valid split if:
        # 1. The predictions are different
        # 2. At least one segment has a register not in the document-level predictions
        if pred1 != pred2 and (pred1 - doc_predictions or pred2 - doc_predictions):
            metrics = {
                "segment1_predictions": list(pred1),
                "segment2_predictions": list(pred2),
                "doc_predictions": list(doc_predictions),
            }
            best_split["split"] = split_indices
            best_split["metrics"] = metrics
            break  # Take the first significant register change

    return best_split


def process_text_recursive(text: str) -> Dict:
    """Process a text recursively, attempting to split it only at major register changes."""
    # Split into sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # If text is too short, return it as a leaf segment
    total_text = " ".join(sentences)
    if len(total_text) < MIN_CHARS * 2:  # Need enough for two segments
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

    # Generate binary splits that meet minimum length requirement
    splits = []
    for i in range(1, len(sentences)):
        left_indices = list(range(i))
        right_indices = list(range(i, len(sentences)))

        # Get text for each potential segment
        left_text = " ".join([sentences[j] for j in left_indices])
        right_text = " ".join([sentences[j] for j in right_indices])

        # Only include split if both segments meet minimum length
        if len(left_text) >= MIN_CHARS and len(right_text) >= MIN_CHARS:
            splits.append((left_indices, right_indices))

    # If no valid splits are found, return as leaf
    if not splits:
        return {
            "text": total_text,
            "sentences": sentences,
            "predictions": get_predictions(total_text),
            "is_leaf": True,
        }

    # Find a significant register change
    best_split = find_register_split(splits, sentences)

    # Only split if we found a significant register change
    if best_split["split"] is None:
        return {
            "text": total_text,
            "sentences": sentences,
            "predictions": get_predictions(total_text),
            "is_leaf": True,
        }

    split_indices = best_split["split"]

    # Get text for each segment
    segment1_text = " ".join([sentences[i] for i in split_indices[0]])
    segment2_text = " ".join([sentences[i] for i in split_indices[1]])

    # Only recursively process segments if they have different registers from parent
    segment1_preds = set(get_predictions(segment1_text))
    segment2_preds = set(get_predictions(segment2_text))
    doc_preds = set(get_predictions(total_text))

    # If both segments have the same registers as the parent, don't split further
    if segment1_preds.issubset(doc_preds) and segment2_preds.issubset(doc_preds):
        segment1_analysis = {
            "text": segment1_text,
            "sentences": [sentences[i] for i in split_indices[0]],
            "predictions": list(segment1_preds),
            "is_leaf": True,
        }
        segment2_analysis = {
            "text": segment2_text,
            "sentences": [sentences[i] for i in split_indices[1]],
            "predictions": list(segment2_preds),
            "is_leaf": True,
        }
    else:
        # Otherwise try to split further
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


def collect_segments(segment: Dict) -> List[Dict]:
    """Helper function to collect only the final segments after all splits."""
    if segment["is_leaf"]:
        # If it's a leaf node, return it as a final segment
        return [
            {
                "text": segment["text"],
                "predictions": segment["predictions"],
                "is_leaf": True,
            }
        ]
    else:
        # If it's not a leaf, only collect segments from its children
        result = []
        result.extend(collect_segments(segment["segments"][0]))
        result.extend(collect_segments(segment["segments"][1]))
        return result


def print_segments(segments: List[Dict]):
    """Helper function to print final segments."""
    print(f"\nTotal final segments: {len(segments)}")
    for i, segment in enumerate(segments, 1):
        print(f"\nSegment {i}:")
        print(f"Length: {len(segment['text'])} chars")
        print(f"Predictions: {', '.join(segment['predictions'])}")
        print(f"Text: {segment['text']}")


def process_tsv_file(input_file_path: str, output_file_path: str):
    """Process texts from TSV file and save recursive segmentations with predictions."""
    df = pd.read_csv(
        input_file_path, sep="\t", header=None, na_values="", keep_default_na=False
    )

    with open(output_file_path, "w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            true_labels = row[0].split()  # Assuming labels are space-separated
            text = row[1]
            text = truncate_text_to_tokens(text)

            # Process the text recursively
            results = process_text_recursive(text)

            # Collect all segments
            all_segments = collect_segments(results)

            print(f"\nDocument {idx}:")
            print(f"True labels: {', '.join(true_labels)}")
            print(f"Document-level predictions: {', '.join(results['predictions'])}")
            print_segments(all_segments)
            print("\n")

            # Add metadata and write to JSONL in flat structure
            output_record = {
                "text_id": idx,
                "true_labels": true_labels,
                "document_predictions": results["predictions"],
                "segments": all_segments,
            }
            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 segment.py input_file.tsv output_file.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_tsv_file(input_file, output_file)
