#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


Stages
------
1) prepare_data   : download + format + split data, write SFT & preference jsonl
2) train_sft      : LoRA / full‑finetune a small causal‑LM on SFT pairs
3) train_reward   : train a reward model from preference comparisons
4) train_ppo      : RLHF with PPOTrainer (uses reward model as critic)
5) evaluate       : quick sentiment‑based empathy proxy + print results

Usage
-----
# one‑click full pipeline
python main.py

# or stage‑by‑stage
python main.py prepare_data --help
python main.py train_sft   --train_file data/sft_train.jsonl ...

Date   : 2025‑05‑08
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import load_dataset, Dataset, DatasetDict
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig
from trl import PPOConfig, PPOTrainer

# ---------- Global config ----------------------------------------------------


@dataclass
class CFG:
    # paths
    DATA_DIR: str = "data"
    CKPT_DIR: str = "checkpoints"
    # models & datasets
    BASE_MODEL: str = "microsoft/DialoGPT-small"
    DATASET: str = "Amod/mental_health_counseling_conversations"
    # training
    EPOCHS: int = 1
    TRAIN_BATCH: int = 4
    LR: float = 1e-5
    # PPO
    PPO_STEPS: int = 64
    PPO_BATCH: int = 4
    PPO_EPOCHS: int = 1
    SEED: int = 42


random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)

Path(CFG.DATA_DIR).mkdir(exist_ok=True)
Path(CFG.CKPT_DIR).mkdir(exist_ok=True)

# ---------- Helper functions -------------------------------------------------


def save_jsonl(records: List[Dict[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ---------- Stage 1 : prepare_data -------------------------------------------


def prepare_data(
    dataset_name: str = CFG.DATASET,
    out_dir: str = CFG.DATA_DIR,
    max_pairs: int | None = None,
) -> None:
    print(f"[prepare_data] Downloading & formatting dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    sft_records: List[Dict[str, str]] = []
    pref_records: List[Dict[str, str]] = []

    for ex in tqdm(ds, desc="Formatting SFT pairs"):
        # 1) conversation list [{"role": "...", "text": "..."}]
        convo = ex.get("conversation")
        if convo and isinstance(convo, list):
            for i in range(len(convo) - 1):
                first_role = convo[i].get("role", "").lower()
                second_role = convo[i + 1].get("role", "").lower()
                if first_role in {"client", "user", "patient"} and second_role in {
                    "therapist",
                    "assistant",
                    "counselor",
                }:
                    prompt = convo[i]["text"].strip()
                    completion = convo[i + 1]["text"].strip()
                    sft_records.append({"prompt": prompt, "completion": completion})

        # 2) question / answer
        elif ex.get("question") and ex.get("answer"):
            sft_records.append(
                {"prompt": ex["question"].strip(), "completion": ex["answer"].strip()}
            )

        # 3) prompt / response
        elif ex.get("prompt") and ex.get("response"):
            sft_records.append(
                {"prompt": ex["prompt"].strip(), "completion": ex["response"].strip()}
            )

        if max_pairs and len(sft_records) >= max_pairs:
            break

    # preference pairs (simple heuristic: duplicate SFT as pos/neg)
    for rec in sft_records:
        pref_records.append(
            {
                "prompt": rec["prompt"],
                "chosen": rec["completion"],
                "rejected": random.choice(sft_records)["completion"],
            }
        )

    out_sft = Path(out_dir) / "sft_train.jsonl"
    out_pref = Path(out_dir) / "prefs.jsonl"
    save_jsonl(sft_records, out_sft)
    save_jsonl(pref_records, out_pref)

    print(
        f"[prepare_data] wrote {len(sft_records):,} SFT samples → {out_sft}\n"
        f"[prepare_data] wrote {len(pref_records):,} preference pairs → {out_pref}"
    )


# ---------- Stage 2 : train_sft ----------------------------------------------


def tokenize_function(example, tokenizer):
    # add EOS token
    return tokenizer(
        example["prompt"] + tokenizer.eos_token + example["completion"],
        truncation=True,
        max_length=512,
    )


def train_sft(
    train_file: str = f"{CFG.DATA_DIR}/sft_train.jsonl",
    output_dir: str = f"{CFG.CKPT_DIR}/sft",
    base_model: str = CFG.BASE_MODEL,
) -> None:
    print(f"[train_sft] Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    ds = Dataset.from_list(load_jsonl(train_file))
    ds = ds.shuffle(seed=CFG.SEED)
    ds_tok = ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=CFG.TRAIN_BATCH,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LR,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tok)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train_sft] checkpoint saved → {output_dir}")


# ---------- Stage 3 : train_reward -------------------------------------------


def tokenize_reward(example, tokenizer):
    return tokenizer(example["prompt"] + tokenizer.eos_token + example["chosen"], truncation=True, max_length=512)


def train_reward(
    pref_file: str = f"{CFG.DATA_DIR}/prefs.jsonl",
    output_dir: str = f"{CFG.CKPT_DIR}/reward",
    base_model: str = CFG.BASE_MODEL,
) -> None:
    print("[train_reward] Training reward model (simple preference --> binary class)")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1
    )

    prefs = load_jsonl(pref_file)
    # label = 1 for chosen, 0 for rejected
    pos_samples = [
        {"text": p["prompt"] + tokenizer.eos_token + p["chosen"], "label": 1}
        for p in prefs
    ]
    neg_samples = [
        {"text": p["prompt"] + tokenizer.eos_token + p["rejected"], "label": 0}
        for p in prefs
    ]
    ds = Dataset.from_list(pos_samples + neg_samples).shuffle(seed=CFG.SEED)
    ds_tok = ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["text"],
    )
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=CFG.TRAIN_BATCH,
        num_train_epochs=CFG.EPOCHS,
        learning_rate=CFG.LR,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train_reward] checkpoint saved → {output_dir}")


# ---------- Stage 4 : train_ppo ----------------------------------------------


def train_ppo(
    sft_ckpt: str = f"{CFG.CKPT_DIR}/sft",
    reward_ckpt: str = f"{CFG.CKPT_DIR}/reward",
    output_dir: str = f"{CFG.CKPT_DIR}/ppo",
    pref_file: str = f"{CFG.DATA_DIR}/prefs.jsonl",
) -> None:
    print("[train_ppo] RLHF with PPO – minimal demo (1 epoch)")
    tokenizer = AutoTokenizer.from_pretrained(sft_ckpt)
    model = AutoModelForCausalLM.from_pretrained(sft_ckpt)
    ref_model = AutoModelForSequenceClassification.from_pretrained(reward_ckpt)

    ppo_config = PPOConfig(
        batch_size=CFG.PPO_BATCH,
        forward_batch_size=CFG.PPO_BATCH,
        ppo_epochs=CFG.PPO_EPOCHS,
        learning_rate=CFG.LR,
    )
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    prefs = load_jsonl(pref_file)[: CFG.PPO_STEPS]
    queries = [p["prompt"] for p in prefs]

    for step, query in enumerate(queries):
        response_tensors = ppo_trainer.generate(query, max_new_tokens=64)
        rewards = torch.tensor(
            [
                ref_model(
                    **tokenizer(
                        query + tokenizer.eos_token + tokenizer.decode(r[0]),
                        return_tensors="pt",
                    )
                ).logits.squeeze()
            ]
        )
        ppo_trainer.step([query], response_tensors, rewards)
        if step % 10 == 0:
            print(f"[train_ppo] step {step}/{len(queries)}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train_ppo] RLHF‑tuned checkpoint → {output_dir}")


# ---------- Stage 5 : evaluate ----------------------------------------------


def evaluate(
    ppo_ckpt: str = f"{CFG.CKPT_DIR}/ppo",
    eval_num: int = 100,
    save_file: str = f"{CFG.DATA_DIR}/eval_metrics.txt",
) -> None:
    print("[evaluate] Quick empathy proxy via sentiment ratio")
    tokenizer = AutoTokenizer.from_pretrained(ppo_ckpt)
    model = AutoModelForCausalLM.from_pretrained(ppo_ckpt)
    sia = SentimentIntensityAnalyzer()

    prompts = [
        "I feel so anxious lately and I don't know why.",
        "I have trouble sleeping every night.",
        "I'm afraid my friends don't really like me.",
        "Some days I just lack the motivation to get out of bed.",
        "I'm overwhelmed with work and study pressures.",
    ] * (eval_num // 5)

    positive = 0
    for p in tqdm(prompts, desc="Generating & scoring"):
        inputs = tokenizer(p, return_tensors="pt")
        out_ids = model.generate(**inputs, max_new_tokens=64)
        response = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        score = sia.polarity_scores(response)["compound"]
        if score >= 0:
            positive += 1

    ratio = positive / len(prompts)
    print(f"[evaluate] Positive‑sentiment responses: {positive}/{len(prompts)}")
    print(f"[evaluate] Empathy ratio = {ratio:.2%}")

    Path(save_file).write_text(f"Empathy_ratio = {ratio:.4f}\n")
    print(f"[evaluate] metrics written → {save_file}")


# ---------- CLI & one‑click ---------------------------------------------------


def parse_cli_and_dispatch() -> None:
    parser = argparse.ArgumentParser(
        description="RLHF pipeline for mental‑health chatbot"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare_data")
    sub.add_parser("train_sft")
    sub.add_parser("train_reward")
    sub.add_parser("train_ppo")
    sub.add_parser("evaluate")

    args = parser.parse_args()

    if args.command == "prepare_data":
        prepare_data()
    elif args.command == "train_sft":
        train_sft()
    elif args.command == "train_reward":
        train_reward()
    elif args.command == "train_ppo":
        train_ppo()
    elif args.command == "evaluate":
        evaluate()
    else:
        raise ValueError(f"Unknown command {args.command}")


def main() -> None:
    if len(sys.argv) == 1:
        # one‑click full pipeline
        print("[one‑click] No arguments detected → running full 5‑stage pipeline\n")
        t0 = time.time()
        prepare_data()
        train_sft()
        train_reward()
        train_ppo()
        evaluate()
        print(
            f"\n[one‑click] Pipeline finished in {time.time() - t0:.1f} s! "
            f"All artifacts in '{CFG.DATA_DIR}' & '{CFG.CKPT_DIR}' folders."
        )
    else:
        parse_cli_and_dispatch()


if __name__ == "__main__":
    main()
