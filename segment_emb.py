import pandas as pd
import numpy as np
import torch
import spacy
import json
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Dict
from tqdm import tqdm

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
    """Find the single best split that optimizes both dissimilarity between segments and similarity within segments."""

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_segment_embedding(indices: List[int]) -> np.ndarray:
        segment_embeddings = embeddings[indices]
        return np.mean(segment_embeddings, axis=0)

    def compute_internal_similarity(indices: List[int]) -> float:
        if len(indices) <= 1:
            return 1.0
        segment_embeddings = embeddings[indices]
        similarities = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                similarities.append(
                    cosine_similarity(segment_embeddings[i], segment_embeddings[j])
                )
        return np.mean(similarities) if similarities else 1.0

    best_split = {"score": -1, "split": None, "metrics": None}

    for split_indices in splits:
        segment1_indices, segment2_indices = split_indices

        segment1_emb = get_segment_embedding(segment1_indices)
        segment2_emb = get_segment_embedding(segment2_indices)

        inter_segment_similarity = cosine_similarity(segment1_emb, segment2_emb)
        inter_segment_dissimilarity = 1 - inter_segment_similarity

        segment1_internal_similarity = compute_internal_similarity(segment1_indices)
        segment2_internal_similarity = compute_internal_similarity(segment2_indices)
        avg_internal_similarity = np.mean(
            [segment1_internal_similarity, segment2_internal_similarity]
        )

        combined_score = inter_segment_dissimilarity * avg_internal_similarity

        metrics = {
            "inter_segment_dissimilarity": float(inter_segment_dissimilarity),
            "avg_internal_similarity": float(avg_internal_similarity),
            "segment1_internal_similarity": float(segment1_internal_similarity),
            "segment2_internal_similarity": float(segment2_internal_similarity),
            "combined_score": float(combined_score),
        }

        if combined_score > best_split["score"]:
            best_split["score"] = combined_score
            best_split["split"] = split_indices
            best_split["metrics"] = metrics

    return best_split


def process_text(text: str) -> Dict:
    """Process a single text and return the optimal segmentation with predictions."""
    # Split into sentences using spaCy
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Check if text is long enough to potentially split
    total_text = " ".join(sentences)
    if len(total_text) < 1000:  # Need at least 1000 chars to make two 500-char segments
        return {
            "error": "Text is too short for meaningful segmentation (needs 1000+ characters)"
        }

    # Check if we have enough sentences
    if len(sentences) < 2:
        return {"error": "Text has fewer than 2 sentences"}

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
        if len(left_text) >= 500 and len(right_text) >= 500:
            splits.append((left_indices, right_indices))

    # Find optimal split
    best_split = find_optimal_split(embeddings, splits)
    split_indices = best_split["split"]

    # Get text for each segment
    segment1_text = " ".join([sentences[i] for i in split_indices[0]])
    segment2_text = " ".join([sentences[i] for i in split_indices[1]])

    # Get predictions for full text and segments
    full_text_predictions = get_predictions(text)
    segment1_predictions = get_predictions(segment1_text)
    segment2_predictions = get_predictions(segment2_text)

    return {
        "sentences": sentences,
        "predictions": {
            "full_text": full_text_predictions,
            "segment1": segment1_predictions,
            "segment2": segment2_predictions,
        },
        "optimal_split": {
            "segment1": {
                "text": segment1_text,
                "sentences": [sentences[i] for i in split_indices[0]],
            },
            "segment2": {
                "text": segment2_text,
                "sentences": [sentences[i] for i in split_indices[1]],
            },
            "indices": split_indices,
            "metrics": best_split["metrics"],
        },
    }


def process_tsv_file(input_file_path: str, output_file_path: str):
    """Process texts from TSV file and save optimal segmentations with predictions."""
    df = pd.read_csv(
        input_file_path, sep="\t", header=None, na_values="", keep_default_na=False
    )

    with open(output_file_path, "w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            true_labels = row[0].split()  # Assuming labels are space-separated
            text = row[1]

            # Process the text
            results = process_text(text)

            if "error" in results:
                print(f"\nDocument {idx}: {results['error']}\n")
            else:
                print(f"\nDocument {idx}:")
                print(
                    f"Document labels: {', '.join(results['predictions']['full_text'])}"
                )
                print(
                    f"Segment 1 labels: {', '.join(results['predictions']['segment1'])}"
                )
                print(
                    f"Segment 1: {results['optimal_split']['segment1']['text'][:100]}..."
                )
                print("----")
                print(
                    f"Segment 2 labels: {', '.join(results['predictions']['segment2'])}"
                )
                print(
                    f"Segment 2: {results['optimal_split']['segment2']['text'][:100]}..."
                )
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
