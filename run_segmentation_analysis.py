from itertools import combinations
import sys

import pandas as pd
import spacy
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import psutil

# At the start of your script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
model_name = "TurkuNLP/web-register-classification-multilingual"
tokenizer_name = "xlm-roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model = model.to(device)
model.eval()
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
    # Move input tensors to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # Move predictions back to CPU before converting to numpy
    probs = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()

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


def combine_short_sentences(sentences, min_words=MIN_WORDS, max_segments=20):
    """
    Combine sentences adaptively by increasing min_words until we get fewer than max_segments.
    """

    def _combine_with_min_words(sentences, min_words):
        if not sentences:
            return []

        combined_sentences = []
        current_segment = []
        current_count = 0

        for sentence in sentences:
            word_count = len(word_tokenize(sentence))

            if current_count + word_count < min_words:
                current_segment.append(sentence)
                current_count += word_count
            else:
                if current_segment:
                    combined_sentences.append(" ".join(current_segment))
                current_segment = [sentence]
                current_count = word_count

        if current_segment:
            combined_sentences.append(" ".join(current_segment))

        return combined_sentences

    # Start with initial min_words
    current_min_words = min_words
    while True:
        combined = _combine_with_min_words(sentences, current_min_words)
        if len(combined) <= max_segments:
            print(
                f"Combined to {len(combined)} segments with min_words={current_min_words}"
            )
            return combined

        # Increase min_words by 1 and try again
        current_min_words += 1
        print(
            f"Too many segments ({len(combined)}), increasing min_words to {current_min_words}"
        )


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


def log_memory_usage():

    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def analyze_hybrid_discreteness(text, true_labels):
    log_memory_usage()
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

    best_partition = None
    max_discreteness = float("-inf")
    best_predictions_true = None
    best_predictions_all = None

    for partition in partitions:
        block_predictions = [get_model_predictions(block) for block in partition]

        discreteness = sum(
            abs(block_pred[pred_ids[0]] - block_pred[pred_ids[1]])
            for block_pred in block_predictions
        ) / len(partition)

        if discreteness > max_discreteness:
            max_discreteness = discreteness
            best_partition = partition
            # Convert numpy arrays to simple lists before storing
            best_predictions_true = [
                block_pred[pred_ids].tolist() for block_pred in block_predictions
            ]
            best_predictions_all = [pred.tolist() for pred in block_predictions]

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

    # Process documents and store results
    results = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analyzing documents"):
        result = analyze_hybrid_discreteness(row["text"], row["labels"])
        if result.get("true_positive"):
            result["id"] = idx
            result["true_labels"] = row["labels"]
            results.append(result)

    # Create DataFrame and save
    df_results = pd.DataFrame(results)

    # Ensure all numpy arrays are converted to lists before saving
    for col in ["best_predictions_true", "best_predictions_all"]:
        if col in df_results.columns:
            df_results[col] = df_results[col].apply(
                lambda x: x if isinstance(x, list) else x.tolist()
            )

    df_results.to_csv(
        f"hybrid_discreteness_results_{file_path.split('.')[0].split('/')[-1]}.csv",
        index=False,
    )
    return results


# Get file name from sys argv
file_name = sys.argv[1]


results = process_tsv_file(file_name)
