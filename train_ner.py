import numpy as np
import warnings
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
import evaluate

warnings.filterwarnings("ignore")


# Step 1. Load Dataset

dataset = load_from_disk("./processed_data_full")
print("Dataset loaded:", dataset)


if "test" not in dataset:
    print("No test split found. Creating test split (10% of train)...")
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": dataset["validation"],
        "test": split_dataset["test"]
    })
    print("Added test split:", dataset)


# Step 2. Tokenizer & Model

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

label_list = ["O", "B-PROBLEM", "I-PROBLEM", "B-TEST", "I-TEST", "B-TREATMENT", "I-TREATMENT"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    trust_remote_code=True,
    use_safetensors=True
)


# Step 3. Tokenize & Align Labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id.get(label[word_idx], 0))
            else:
                label_ids.append(label_to_id.get(label[word_idx].replace("B-", "I-"), 0))
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
print("Tokenization done.")



example = tokenized_datasets["train"][0]
tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
labels = [id_to_label.get(l, "IGN") for l in example["labels"][:30]]
print("\n🔍 Token-label alignment check (first 30 tokens):")
for t, l in zip(tokens[:30], labels):
    print(f"{t:15s} -> {l}")


# Step 4. Metrics

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id_to_label[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    detailed = {}
    for entity_type, scores in results.items():
        if isinstance(scores, dict) and "precision" in scores:
            detailed[f"{entity_type}_precision"] = scores["precision"]
            detailed[f"{entity_type}_recall"] = scores["recall"]
            detailed[f"{entity_type}_f1"] = scores["f1"]

    overall = {
        "precision": results.get("overall_precision", 0.0),
        "recall": results.get("overall_recall", 0.0),
        "f1": results.get("overall_f1", 0.0),
        "accuracy": results.get("overall_accuracy", 0.0),
    }
    return {**overall, **detailed}


# Step 5. Training Arguments

try:
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  
        learning_rate=5e-5,           
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=15,       
        weight_decay=0.01,
        warmup_ratio=0.05,           
        logging_dir="./logs",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none"
    )
except TypeError:
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",       
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=30,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch",
        warmup_ratio=0.05,
        save_total_limit=2,
        report_to="none"
    )


# Step 6. Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train()
trainer.save_model("./clinicalbert_ner_model")
print("\nModel training finished and saved to ./clinicalbert_ner_model")


# Step 8. Evaluate on Validation and Test Sets

print("\nValidation Set Evaluation:")
val_metrics = trainer.evaluate(tokenized_datasets["validation"])
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\nTest Set Evaluation:")
test_metrics = trainer.evaluate(tokenized_datasets["test"])
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\nAll evaluations complete.")