import os
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from collections import Counter


base_dir = "/Users/rosie/Desktop/n2c2/concept_assertion_relation_training_data"
output_dir = "./processed_data_full"
os.makedirs(output_dir, exist_ok=True)


def read_concept_file(concept_path):
    entities = []
    with open(concept_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            
            match = re.search(r'c="(.+?)"\s+(\d+):(\d+)\s+(\d+):(\d+)\|\|t="(\w+)"', line)
            if match:
                ent_text = match.group(1)
                line_start = int(match.group(2))
                tok_start = int(match.group(3))
                line_end = int(match.group(4))
                tok_end = int(match.group(5))
                label = match.group(6).upper()
                entities.append((line_start, tok_start, line_end, tok_end, ent_text, label))
    return entities


def tokenize_and_label_by_line(text_lines, entities):
    tokens_all, labels_all = [], []

    for line_idx, line in enumerate(text_lines):
        tokens = line.strip().split()
        labels = ["O"] * len(tokens)

        for ent in entities:
            line_start, tok_start, line_end, tok_end, ent_text, label = ent
            if line_idx == line_start == line_end:
                for i in range(tok_start, tok_end + 1):
                    if i < len(labels):
                        labels[i] = f"I-{label}" if i > tok_start else f"B-{label}"

        if tokens:
            tokens_all.extend(tokens)
            labels_all.extend(labels)

    return tokens_all, labels_all


def load_i2b2_data():
    data = {"tokens": [], "labels": []}
    total_files, annotated = 0, 0

    for site in ["beth", "partners"]:
        txt_dir = os.path.join(base_dir, site, "txt")
        concept_dir = os.path.join(base_dir, site, "concept")

        for filename in os.listdir(txt_dir):
            if not filename.endswith(".txt"):
                continue
            total_files += 1
            file_id = filename.replace(".txt", "")
            txt_path = os.path.join(txt_dir, filename)
            con_path = os.path.join(concept_dir, f"{file_id}.con")

            if not os.path.exists(con_path):
                continue
            annotated += 1

            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text_lines = f.readlines()

            entities = read_concept_file(con_path)
            tokens, labels = tokenize_and_label_by_line(text_lines, entities)
            data["tokens"].append(tokens)
            data["labels"].append(labels)

    print(f"Total text files: {total_files}")
    print(f"Annotated files used: {annotated}")
    return data


print("Loading data from beth & partners ...")
data = load_i2b2_data()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    data["tokens"], data["labels"], test_size=0.2, random_state=42
)

train_dataset = Dataset.from_dict({"tokens": train_texts, "ner_tags": train_labels})
val_dataset = Dataset.from_dict({"tokens": val_texts, "ner_tags": val_labels})
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
dataset.save_to_disk(output_dir)
print(f"\nSaved HuggingFace Dataset to: {output_dir}")


print(f"\nTrain samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

example = train_dataset[0]
print("\nExample tokens:", example["tokens"][:30])
print("Example labels:", example["ner_tags"][:30])


all_labels = [label for seq in data["labels"] for label in seq]
counter = Counter(all_labels)
print("\nLabel distribution:")
for k, v in counter.items():
    print(f"{k:<12} : {v}")