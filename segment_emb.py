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


def find_optimal_split(
    embeddings: np.ndarray, splits: List[Tuple[List[int], List[int]]]
) -> Dict:
    """Find the single best split by comparing against the overall document structure."""

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_segment_embedding(indices: List[int]) -> np.ndarray:
        segment_embeddings = embeddings[indices]
        return np.mean(segment_embeddings, axis=0)

    def compute_split_score(
        split_indices: Tuple[List[int], List[int]]
    ) -> Tuple[float, Dict]:
        segment1_indices, segment2_indices = split_indices

        # Get embeddings for each segment and full document
        segment1_emb = get_segment_embedding(segment1_indices)
        segment2_emb = get_segment_embedding(segment2_indices)
        full_doc_emb = get_segment_embedding(list(range(len(embeddings))))

        # Get segment-to-segment difference
        inter_segment_similarity = cosine_similarity(segment1_emb, segment2_emb)

        # Get segment-to-document differences
        seg1_doc_sim = cosine_similarity(segment1_emb, full_doc_emb)
        seg2_doc_sim = cosine_similarity(segment2_emb, full_doc_emb)

        # If segments are more similar to each other than to the document average,
        # this is probably not a real split point
        if inter_segment_similarity > (seg1_doc_sim + seg2_doc_sim) / 2:
            return 0.0, {}

        # Score based on how different the segments are
        split_score = 1 - inter_segment_similarity

        metrics = {
            "inter_segment_similarity": float(inter_segment_similarity),
            "seg1_doc_similarity": float(seg1_doc_sim),
            "seg2_doc_similarity": float(seg2_doc_sim),
            "split_score": float(split_score),
        }

        return split_score, metrics

    # Try all possible splits
    split_scores = []
    for split in splits:
        score, metrics = compute_split_score(split)
        if score > 0:  # Only consider splits that passed our basic check
            split_scores.append((score, split, metrics))

    # If no splits passed our basic quality check, return no split
    if not split_scores:
        return {"score": -1, "split": None, "metrics": None}

    # Find the best split among qualifying ones
    best_score, best_split, best_metrics = max(split_scores, key=lambda x: x[0])

    return {"score": best_score, "split": best_split, "metrics": best_metrics}


def process_text_recursive(text: str) -> Dict:
    """Process a text recursively, attempting to split it and its resulting segments."""
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

    # Find optimal split
    best_split = find_optimal_split(embeddings, splits)

    # Only split if we found a significant split
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
