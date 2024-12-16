from itertools import combinations

import pandas as pd
import spacy
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load models
model_name = "TurkuNLP/web-register-classification-multilingual"
tokenizer_name = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Get label mapping - only first 9 labels
label_names = {i: model.config.id2label[i] for i in range(9)}
label_to_id = {v: k for k, v in label_names.items()}

# Load spaCy
nlp = spacy.load("xx_ent_wiki_sm")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Default settings
threshold = 0.475
MIN_WORDS = 15
MAX_TOKENS = 512

prediction_cache = {}


def get_model_predictions(text):
    if text in prediction_cache:
        return prediction_cache[text]

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_TOKENS
    )
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).squeeze().detach().numpy()

    prediction_cache[text] = probs
    return probs


def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def truncate_text_to_tokens(text):
    """Truncate text to fit within model's token limit"""
    tokens = tokenizer(text, truncation=True, max_length=MAX_TOKENS)
    truncated_text = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
    return truncated_text


def combine_short_sentences(sentences, min_words=MIN_WORDS):
    """Combine sentences to ensure no segment has fewer than min_words words."""
    if not sentences:
        return []

    combined_sentences = []
    current_segment = []
    current_count = 0

    for sentence in sentences:
        word_count = len(word_tokenize(sentence))

        # Add to the current segment if it's under the limit
        if current_count + word_count < min_words:
            current_segment.append(sentence)
            current_count += word_count
        else:
            # If current segment has enough words, start a new one
            if current_segment:
                combined_sentences.append(" ".join(current_segment))
            current_segment = [sentence]
            current_count = word_count

    # Add the last segment
    if current_segment:
        combined_sentences.append(" ".join(current_segment))

    return combined_sentences


def generate_ordered_partitions(sentences):
    """Generate all possible ordered partitions of sentences."""
    n = len(sentences)

    def partition_from_indices(indices):
        """Convert partition indices to actual sentence groups."""
        result = []
        prev = 0
        for index in indices:
            result.append(" ".join(sentences[prev:index]))
            prev = index
        result.append(" ".join(sentences[prev:]))
        return result

    all_partitions = []
    # Generate all possible combinations of partition points
    for r in range(n):  # r is the number of partition points
        for indices in combinations(
            range(1, n), r
        ):  # Partition points are between 1 and n-1
            all_partitions.append(partition_from_indices(indices))

    return all_partitions


def analyze_hybrid_discreteness(text, true_labels):
    print("-" * 50)
    prediction_cache.clear()

    # First truncate the text
    truncated_text = truncate_text_to_tokens(text)

    probs = get_model_predictions(truncated_text)

    pred_ids = [i for i, p in enumerate(probs[:9]) if p > threshold]
    pred_labels = [label_names[label_id] for label_id in pred_ids]
    # Check that set of label names is same as true labels
    if not set(true_labels) == set(pred_labels):
        return {"true_positive": False}

    print(f"\nAnalyzing text with true labels: {true_labels}")
    print("Predicted labels:")
    for label_id in pred_ids:
        print(f"- {label_names[label_id]} (probability: {probs[label_id]:.3f})")

    initial_sentences = split_into_sentences(truncated_text)
    sentences = combine_short_sentences(initial_sentences)
    print(f"\nSentence processing:")
    print(f"Initial sentences: {len(initial_sentences)}")
    print(f"After combining short sentences: {len(sentences)}")

    partitions = generate_ordered_partitions(sentences)
    print(f"Generated {len(partitions)} possible partitions")

    # Initialize variables to track the best partition
    best_partition = None
    max_discreteness = float("-inf")
    best_predictions_true = None  # To store predictions for the best partition
    best_predictions_all = None  # To store predictions for the best partition

    for partition in partitions:
        # Get predictions for each block in the partition
        block_predictions = [get_model_predictions(block) for block in partition]

        # Compute discreteness for the current partition
        discreteness = sum(
            abs(block_pred[pred_ids[0]] - block_pred[pred_ids[1]])
            for block_pred in block_predictions
        ) / len(partition)

        # Update the best partition if this one has higher discreteness
        if discreteness > max_discreteness:
            max_discreteness = discreteness
            best_partition = partition

            # get the block predictions for this partition for the 2 labels
            best_predictions_true = [
                block_pred[pred_ids] for block_pred in block_predictions
            ]
            best_predictions_all = block_predictions

    return {
        "true_positive": True,
        "best_partition": best_partition,
        "max_discreteness": max_discreteness,
        "best_predictions_true": best_predictions_true,
        "best_predictions_all": best_predictions_all,
    }


def process_tsv_file(file_path):
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["labels", "text"],
        na_values="",
        keep_default_na=False,
    )

    # Filter and preprocess labels
    df["labels"] = df["labels"].apply(
        lambda x: [label for label in x.split() if label in label_to_id]
    )
    df = df[df["labels"].apply(len) == 2]

    print(df)

    # Enable tqdm for pandas DataFrame iteration
    results = [
        {
            **analyze_hybrid_discreteness(row["text"], row["labels"]),
            "id": idx,
            "true_labels": row["labels"],
        }
        for idx, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Analyzing documents"
        )
    ]

    # Filter results to retain only where true_positive = True
    results = [result for result in results if result.get("true_positive")]
    df_results = pd.DataFrame(results)
    df_results.to_csv("hybrid_discreteness_results.csv", index=False)


results = process_tsv_file("fi_all.tsv")
