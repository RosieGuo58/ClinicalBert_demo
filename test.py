import os
import re
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import classification_report


def parse_con_file(con_path: str) -> List[Tuple[int, int, int, int, str]]:
    """Parse a .con file and return a list of spans (start_line, start_idx, end_line, end_idx, label)."""
    entities = []
    with open(con_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            label_match = re.search(r't="([^"]+)"', raw)
            span_matches = re.findall(r'(\d+:\d+)', raw)
            if not label_match or not span_matches:
                continue
            label = label_match.group(1)
            start_line, start_idx = map(int, span_matches[0].split(":"))
            if len(span_matches) > 1:
                end_line, end_idx = map(int, span_matches[1].split(":"))
            else:
                end_line, end_idx = start_line, start_idx
            entities.append((start_line, start_idx, end_line, end_idx, label))
    return entities


def find_n2c2_root(root_dir: str = ".") -> str:
    """Locate the n2c2 directory that contains concept_assertion_relation_training_data."""
    for entry in os.listdir(root_dir):
        candidate = os.path.join(root_dir, entry)
        if not os.path.isdir(candidate):
            continue
        if entry.lower().strip() == "n2c2":
            concept_dir = os.path.join(candidate, "concept_assertion_relation_training_data")
            if os.path.isdir(concept_dir):
                return candidate
    raise FileNotFoundError(
        "n2c2 directory not found. Ensure it is placed next to this script and contains concept_assertion_relation_training_data."
    )


def convert_folder(txt_dir: str, con_dir: str) -> List[Tuple[List[str], List[str]]]:
    """Convert paired txt/con files into line-aligned (tokens, tags) samples."""
    samples: List[Tuple[List[str], List[str]]] = []
    for filename in os.listdir(txt_dir):
        if not filename.endswith(".txt"):
            continue
        txt_path = os.path.join(txt_dir, filename)
        con_path = os.path.join(con_dir, filename.replace(".txt", ".con"))
        if not os.path.exists(con_path):
            continue

        with open(txt_path, "r") as f:
            lines = [line.rstrip("\n") for line in f]

        tokenized_lines = [line.strip().split() for line in lines]
        entities = parse_con_file(con_path)

        start_positions: Dict[Tuple[int, int], str] = {}
        all_positions: Dict[Tuple[int, int], str] = {}
        for s_line, s_idx, e_line, e_idx, label in entities:
            label_upper = label.upper()
            first_coord = None
            for line_id in range(s_line, e_line + 1):
                if line_id >= len(tokenized_lines):
                    break
                tokens = tokenized_lines[line_id]
                if not tokens:
                    continue
                start_idx = s_idx if line_id == s_line else 0
                end_idx = e_idx if line_id == e_line else len(tokens) - 1
                if start_idx >= len(tokens):
                    continue
                end_idx = min(end_idx, len(tokens) - 1)
                if end_idx < start_idx:
                    continue
                if first_coord is None:
                    first_coord = (line_id, start_idx)
                for idx in range(start_idx, end_idx + 1):
                    all_positions[(line_id, idx)] = label_upper
            if first_coord:
                start_positions[first_coord] = label_upper

        for line_id, tokens in enumerate(tokenized_lines):
            if not tokens:
                continue
            tags = []
            for idx, _ in enumerate(tokens):
                if (line_id, idx) in all_positions:
                    current_label = all_positions[(line_id, idx)]
                    if (line_id, idx) in start_positions:
                        tags.append(f"B-{current_label}")
                    else:
                        tags.append(f"I-{current_label}")
                else:
                    tags.append("O")
            samples.append((tokens, tags))
    return samples


def write_bio_file(samples: List[Tuple[List[str], List[str]]], path: str) -> None:
    with open(path, "w") as f:
        for tokens, labels in samples:
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")


def prepare_bio_data() -> None:
    n2c2_root = find_n2c2_root()
    base_dir = os.path.join(n2c2_root, "concept_assertion_relation_training_data")
    samples: List[Tuple[List[str], List[str]]] = []
    for site in ["beth", "partners"]:
        txt_dir = os.path.join(base_dir, site, "txt")
        con_dir = os.path.join(base_dir, site, "concept")
        samples.extend(convert_folder(txt_dir, con_dir))

    random.shuffle(samples)
    train, temp = train_test_split(samples, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    os.makedirs("bio_data", exist_ok=True)
    write_bio_file(train, "bio_data/train.txt")
    write_bio_file(val, "bio_data/dev.txt")
    write_bio_file(test, "bio_data/test.txt")
    print("BIO data generated in bio_data/ folder.")


def read_conll(path: str) -> Dict[str, List[List[str]]]:
    tokens, tags, current_tokens, current_tags = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    tokens.append(current_tokens)
                    tags.append(current_tags)
                    current_tokens, current_tags = [], []
                continue
            token, tag = line.split("\t")
            current_tokens.append(token)
            current_tags.append(tag)
    if current_tokens:
        tokens.append(current_tokens)
        tags.append(current_tags)
    return {"tokens": tokens, "ner_tags": tags}


def train_ner_model() -> None:
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    metric = evaluate.load("seqeval")

    print("Loading BIO data...")
    for split in ["train", "dev", "test"]:
        path = os.path.join("bio_data", f"{split}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run prepare_bio_data() before training.")

    train_data = read_conll("bio_data/train.txt")
    val_data = read_conll("bio_data/dev.txt")
    test_data = read_conll("bio_data/test.txt")

    unique_tags = sorted({tag for doc in train_data["ner_tags"] for tag in doc})
    tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    def tokenize_and_align_labels(examples: Dict[str, List[List[str]]]) -> Dict[str, List[List[int]]]:
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
        )
        aligned_labels = []
        for batch_index, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(tag2id[labels[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    train_ds = Dataset.from_dict(train_data).map(tokenize_and_align_labels, batched=True)
    val_ds = Dataset.from_dict(val_data).map(tokenize_and_align_labels, batched=True)
    test_ds = Dataset.from_dict(test_data).map(tokenize_and_align_labels, batched=True)

    cols_to_remove = [col for col in ["tokens", "ner_tags"] if col in train_ds.column_names]
    train_ds = train_ds.remove_columns(cols_to_remove)
    val_ds = val_ds.remove_columns(cols_to_remove)
    test_ds = test_ds.remove_columns(cols_to_remove)

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print("Detected Apple Silicon GPU (MPS). Training will run on MPS.")
    elif torch.cuda.is_available():
        print("Detected CUDA GPU. Training will run on CUDA.")
    else:
        print("No GPU detected. Training will run on CPU (slow).")

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(unique_tags),
        id2label=id2tag,
        label2id=tag2id,
    )

    args = TrainingArguments(
        output_dir="./ner_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_preds = [
            [id2tag[pred] for pred, label in zip(prediction, label_row) if label != -100]
            for prediction, label_row in zip(predictions, labels)
        ]
        true_labels = [
            [id2tag[label] for pred, label in zip(prediction, label_row) if label != -100]
            for prediction, label_row in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training ClinicalBERT NER model...")
    trainer.train()

    print("Evaluating on test set...")
    results = trainer.predict(test_ds)
    predictions = np.argmax(results.predictions, axis=2)

    true_preds = [
        [id2tag[pred] for pred, label in zip(pred_row, label_row) if label != -100]
        for pred_row, label_row in zip(predictions, results.label_ids)
    ]
    true_labels = [
        [id2tag[label] for pred, label in zip(pred_row, label_row) if label != -100]
        for pred_row, label_row in zip(predictions, results.label_ids)
    ]

    summary = metric.compute(predictions=true_preds, references=true_labels)
    print("\nOverall Results:")
    print(f"Precision: {summary['overall_precision']:.4f}")
    print(f"Recall:    {summary['overall_recall']:.4f}")
    print(f"F1:        {summary['overall_f1']:.4f}")

    report = classification_report(true_labels, true_preds, digits=4)
    print("\nDetailed classification report:")
    print(report)

    with open("ner_detailed_report.txt", "w") as f:
        f.write("=== Overall Results ===\n")
        f.write(f"Precision: {summary['overall_precision']:.4f}\n")
        f.write(f"Recall:    {summary['overall_recall']:.4f}\n")
        f.write(f"F1:        {summary['overall_f1']:.4f}\n\n")
        f.write("=== Per-label Report ===\n")
        f.write(report)

    trainer.save_model("./clinicalbert_n2c2_ner_model")
    tokenizer.save_pretrained("./clinicalbert_n2c2_ner_model")
    print("\nModel and report saved successfully!")


if __name__ == "__main__":
    prepare_bio_data()
    train_ner_model()
