import os
import re
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

BASE_DIR = "/Users/rosie/Desktop/n2c2/concept_assertion_relation_training_data/beth"
TEXT_DIR = os.path.join(BASE_DIR, "txt")
CON_DIR = os.path.join(BASE_DIR, "concept")

def load_annotations(con_path):
    anns = []
    with open(con_path, "r") as f:
        for line in f:
            m = re.match(r'c="(.+?)"\s+(\d+):(\d+)\s+(\d+):(\d+)\|\|t="(.+?)"', line.strip())
            if not m:
                continue
            text, start_line, start_tok, end_line, end_tok, label = m.groups()
            anns.append({
                "text": text,
                "start_line": int(start_line) - 1,  # i2b2行号从1开始
                "start_tok": int(start_tok),
                "end_line": int(end_line) - 1,
                "end_tok": int(end_tok),
                "label": label.upper()
            })
    return anns

def tag_file(txt_path, con_path):
    with open(txt_path, "r") as f:
        lines = [line.strip().split() for line in f.readlines()]
    tags = [["O"] * len(tokens) for tokens in lines]

    anns = load_annotations(con_path)
    for ann in anns:
        for line_id in range(ann["start_line"], ann["end_line"] + 1):
            if line_id < 0 or line_id >= len(lines):
                continue
            start_tok = ann["start_tok"] if line_id == ann["start_line"] else 0
            end_tok = ann["end_tok"] if line_id == ann["end_line"] else len(lines[line_id]) - 1
            for tok_id in range(start_tok, end_tok + 1):
                if tok_id < len(tags[line_id]):
                    if tok_id == start_tok:
                        tags[line_id][tok_id] = f"B-{ann['label']}"
                    else:
                        tags[line_id][tok_id] = f"I-{ann['label']}"
    
    tokens = [t for line in lines for t in line]
    ner_tags = [t for line in tags for t in line]
    return tokens, ner_tags

data = {"tokens": [], "ner_tags": []}

print("📂 Aligning tokens by line and word index ...")

for file in tqdm(os.listdir(TEXT_DIR)):
    if not file.endswith(".txt"):
        continue
    txt_path = os.path.join(TEXT_DIR, file)
    con_path = os.path.join(CON_DIR, file.replace(".txt", ".con"))
    if not os.path.exists(con_path):
        continue

    tokens, tags = tag_file(txt_path, con_path)
    data["tokens"].append(tokens)
    data["ner_tags"].append(tags)

train_tokens, val_tokens, train_tags, val_tags = train_test_split(
    data["tokens"], data["ner_tags"], test_size=0.2, random_state=42
)

train_dataset = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_tags})
val_dataset = Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_tags})
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
dataset.save_to_disk("./processed_data")

all_labels = [t for tags in data["ner_tags"] for t in tags]
print("🧾 Label distribution:", Counter(all_labels))