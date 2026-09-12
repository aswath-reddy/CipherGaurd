"""
DeBERTa Multi-Label Fine-tuning Script for CipherGuard.

Fine-tunes protectai/deberta-v3-base-prompt-injection-v2 with a 5-category
multi-label sigmoid head on the CipherGuard expanded dataset.

Requirements (GPU system):
    pip install transformers>=4.30 torch>=2.0 datasets>=2.14 scikit-learn
    pip install peft>=0.6 accelerate>=0.24   # for LoRA (recommended for <=8GB VRAM)

Usage:
    # Full fine-tuning (requires ~12GB VRAM):
    python -m evaluation.finetune_deberta --mode full

    # LoRA fine-tuning (requires ~4GB VRAM, recommended):
    python -m evaluation.finetune_deberta --mode lora

Output:
    models/cipherguard-deberta-finetuned/
    (Copy this directory to your CPU system and it will be auto-loaded by CipherGuard)
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict

# Allow running from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

CATEGORIES = ["Jailbreak", "Prompt Injection", "PII Leakage", "Malicious Tools", "Hate/Toxicity"]
BASE_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
OUTPUT_DIR = "models/cipherguard-deberta-finetuned"
DATA_PATH = "data/expanded_dataset.json"


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    label_matrix = []
    for d in data:
        if "labels" in d:
            row = [float(d["labels"].get(cat, 0)) for cat in CATEGORIES]
        else:
            # Legacy binary: map to injection categories
            val = float(d.get("label", 0))
            row = [val, val, 0.0, 0.0, 0.0]
        label_matrix.append(row)
    return texts, label_matrix


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    import torch
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    f1_per_cat = {}
    for i, cat in enumerate(CATEGORIES):
        f1 = f1_score(labels[:, i].astype(int), preds[:, i], zero_division=0)
        f1_per_cat[f"f1_{cat.replace('/', '_').replace(' ', '_')}"] = f1
    f1_per_cat["f1_macro"] = np.mean(list(f1_per_cat.values()))
    return f1_per_cat


def run_finetuning(mode: str = "lora"):
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer
        )
        from torch.utils.data import Dataset
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install transformers torch datasets scikit-learn")
        sys.exit(1)

    print("=" * 65)
    print(f"CipherGuard DeBERTa Fine-tuning — mode={mode}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run: python -m evaluation.build_expanded_dataset")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow. Use a GPU system.")

    # Load data
    texts, label_matrix = load_data(DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, label_matrix, test_size=0.15, random_state=42
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    class CipherGuardDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                texts, padding=True, truncation=True, max_length=512
            )
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

    train_dataset = CipherGuardDataset(X_train, y_train)
    val_dataset = CipherGuardDataset(X_val, y_val)

    # Load base model — replace classification head with 5-label multi-label head
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CATEGORIES),
        ignore_mismatched_sizes=True,  # Replace the 2-label head with 5-label head
        problem_type="multi_label_classification"
    )

    # Apply LoRA if requested
    if mode == "lora":
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,
                lora_alpha=32,
                target_modules=["query_proj", "value_proj"],
                lora_dropout=0.1,
                bias="none"
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print("[LoRA] LoRA adapters applied successfully.")
        except ImportError:
            print("[LoRA] peft not installed. Falling back to full fine-tuning.")
            print("  Install with: pip install peft")
            mode = "full"

    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16 if device == "cuda" else 8,
        per_device_eval_batch_size=32 if device == "cuda" else 8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=(device == "cuda"),   # Mixed precision on GPU only
        dataloader_num_workers=0,
        report_to="none",
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    # Merge LoRA weights before saving (if applicable)
    if mode == "lora":
        try:
            model = model.merge_and_unload()
            print("[LoRA] Weights merged successfully.")
        except Exception as e:
            print(f"[LoRA] Could not merge weights: {e}. Saving adapter-only model.")

    # Save full model + tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save category mapping for inference
    config_extra = {"id2label": {str(i): cat for i, cat in enumerate(CATEGORIES)},
                    "label2id": {cat: i for i, cat in enumerate(CATEGORIES)}}
    with open(os.path.join(OUTPUT_DIR, "cipherguard_label_config.json"), "w") as f:
        json.dump(config_extra, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Fine-tuned model saved to: {OUTPUT_DIR}/")
    print("Copy this directory to your CPU system (cipherguard/models/)")
    print("CipherGuard will auto-detect and load it on next startup.")
    print("=" * 65)

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        if "f1" in k:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for CipherGuard")
    parser.add_argument(
        "--mode", choices=["full", "lora"], default="lora",
        help="'lora' (4GB VRAM, recommended) or 'full' (12GB VRAM)"
    )
    args = parser.parse_args()
    run_finetuning(mode=args.mode)
