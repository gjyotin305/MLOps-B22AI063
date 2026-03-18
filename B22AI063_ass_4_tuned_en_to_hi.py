#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import math
import os
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

try:
    from ray import train, tune
    from ray.air import RunConfig
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    RAY_AVAILABLE = True
except Exception:
    RAY_AVAILABLE = False


SEED = 42
MAX_LEN = 50
D_MODEL = 512
NUM_LAYERS = 6
BASELINE_EPOCHS = 100
TRIAL_MAX_EPOCHS = 10
DATA_PATH = "English-Hindi.tsv"
RUNS_DIR = Path("runs_en_hi")
CHECKPOINT_PATH = RUNS_DIR / "baseline_checkpoint.pt"
BASELINE_MODEL_PATH = RUNS_DIR / "baseline_model.pth"
BEST_MODEL_PATH = RUNS_DIR / "best_tuned_model.pth"
EN_VOCAB_PATH = RUNS_DIR / "en_vocab.pkl"
HI_VOCAB_PATH = RUNS_DIR / "hi_vocab.pkl"
BEST_CONFIG_PATH = RUNS_DIR / "best_config.json"
HPO_SUMMARY_PATH = RUNS_DIR / "hpo_summary.json"
HPO_REPORT_PATH = RUNS_DIR / "hpo_report.md"

SMOOTHIE = SmoothingFunction().method4

VAL_DATASET = [
    ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
    ("How are you?", "आप कैसे हैं?"),
    ("You should sleep.", "आपको सोना चाहिए।"),
    ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
    ("Let me tell Tom.", "मुझे टॉम को बताने दीजिए।"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_runs_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["id1", "en", "id2", "hi"])
    df = df[["en", "hi"]].dropna().reset_index(drop=True)
    return df


class Vocabulary:
    def __init__(self, freq_threshold: int = 2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx = 4

    def tokenize(self, sentence: str) -> List[str]:
        return sentence.lower().strip().split()

    def build_vocab(self, sentence_list: Iterable[str]) -> None:
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def numericalize(self, sentence: str) -> List[int]:
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in self.tokenize(sentence)]

    def __len__(self) -> int:
        return len(self.stoi)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])


def encode_sentence(sentence: str, vocab: Vocabulary, max_len: int = MAX_LEN) -> List[int]:
    tokens = [vocab.stoi["<sos>"]] + vocab.numericalize(sentence)[: max_len - 2] + [vocab.stoi["<eos>"]]
    return tokens + [vocab.stoi["<pad>"]] * (max_len - len(tokens))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = q.size(0)
        Q = self.query_linear(q)
        K = self.key_linear(k)
        V = self.value_linear(v)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_weights), V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attention_output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(input_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.dropout(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(target_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.dropout(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = D_MODEL,
        num_layers: int = NUM_LAYERS,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = MAX_LEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_pad_mask(self, seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_subsequent_mask(self, size: int) -> torch.Tensor:
        return torch.tril(torch.ones((size, size), device=next(self.parameters()).device)).bool()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_pad_idx: int, tgt_pad_idx: int) -> torch.Tensor:
        src_mask = self.make_pad_mask(src, src_pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
        tgt_sub_mask = self.make_subsequent_mask(tgt.size(1))
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)


class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, en_vocab: Vocabulary, hi_vocab: Vocabulary, max_len: int = MAX_LEN):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.en_sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    tgt_input = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output


def build_datasets(
    df: pd.DataFrame,
    en_vocab: Vocabulary,
    hi_vocab: Vocabulary,
    max_len: int,
    val_ratio: float = 0.2,
    seed: int = SEED,
) -> Tuple[TranslationDataset, Subset, Subset]:
    full_dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=max_len)
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    return full_dataset, train_subset, val_subset


def make_loaders(train_ds: Subset, val_ds: Subset, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_config(config: Dict) -> None:
    if D_MODEL % int(config["num_heads"]) != 0:
        raise ValueError(f"Invalid config: D_MODEL ({D_MODEL}) not divisible by num_heads ({config['num_heads']}).")


def build_model_from_config(config: Dict, src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    validate_config(config)
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        max_len=MAX_LEN,
        dropout=float(config["dropout"]),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
) -> float:
    model.train()
    running_loss = 0.0

    for src, tgt_input, tgt_output in loader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        output = model(src, tgt_input, src_pad_idx, tgt_pad_idx)
        loss = criterion(output.view(-1, output.shape[-1]), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
) -> float:
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for src, tgt_input, tgt_output in loader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            output = model(src, tgt_input, src_pad_idx, tgt_pad_idx)
            loss = criterion(output.view(-1, output.shape[-1]), tgt_output.reshape(-1))
            running_loss += loss.item()

    return running_loss / max(1, len(loader))


def save_checkpoint(epoch: int, model: nn.Module, optimizer: optim.Optimizer, loss: float, path: Path) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        str(path),
    )


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: Path) -> int:
    if not path.exists():
        return 0

    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load(str(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"])


def translate_sentence(
    model: nn.Module,
    sentence: str,
    en_vocab: Vocabulary,
    hi_vocab: Vocabulary,
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
    max_len: int = MAX_LEN,
) -> str:
    model.eval()
    tokens = encode_sentence(sentence, en_vocab, max_len=max_len)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    tgt_tokens = [hi_vocab["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_pad_idx, tgt_pad_idx)
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab["<eos>"]:
            break

    translated = [hi_vocab.itos[idx] for idx in tgt_tokens[1:-1] if idx in hi_vocab.itos]
    return " ".join(translated)


def evaluate_bleu_nltk(
    model: nn.Module,
    eval_pairs: List[Tuple[str, str]],
    en_vocab: Vocabulary,
    hi_vocab: Vocabulary,
    device: torch.device,
    src_pad_idx: int,
    tgt_pad_idx: int,
    max_len: int = MAX_LEN,
) -> float:
    references = []
    hypotheses = []

    for en_sentence, hi_sentence in eval_pairs:
        pred = translate_sentence(model, en_sentence, en_vocab, hi_vocab, device, src_pad_idx, tgt_pad_idx, max_len)
        references.append([hi_sentence.split()])
        hypotheses.append(pred.split())

    return corpus_bleu(references, hypotheses, smoothing_function=SMOOTHIE)


def train_baseline(
    train_ds: Subset,
    val_ds: Subset,
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_pad_idx: int,
    tgt_pad_idx: int,
    device: torch.device,
    epochs: int = BASELINE_EPOCHS,
    batch_size: int = 60,
    resume: bool = True,
) -> Tuple[nn.Module, float]:
    config = {"num_heads": 8, "d_ff": 2048, "dropout": 0.1, "lr": 1e-4}
    model = build_model_from_config(config, src_vocab_size, tgt_vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]))

    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=batch_size)

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_PATH)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, src_pad_idx, tgt_pad_idx)
        val_loss = evaluate_loss(model, val_loader, criterion, device, src_pad_idx, tgt_pad_idx)
        print(f"[Baseline] Epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        save_checkpoint(epoch + 1, model, optimizer, val_loss, CHECKPOINT_PATH)

    torch.save(model.state_dict(), BASELINE_MODEL_PATH)
    return model, float(val_loss)


def train_best_config(
    best_config: Dict,
    train_ds: Subset,
    val_ds: Subset,
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_pad_idx: int,
    tgt_pad_idx: int,
    device: torch.device,
    epochs: int = TRIAL_MAX_EPOCHS,
) -> Tuple[nn.Module, float]:
    model = build_model_from_config(best_config, src_vocab_size, tgt_vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=float(best_config["lr"]))
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=int(best_config["batch_size"]))

    val_loss = float("inf")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, src_pad_idx, tgt_pad_idx)
        val_loss = evaluate_loss(model, val_loader, criterion, device, src_pad_idx, tgt_pad_idx)
        print(f"[BestConfig] Epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), BEST_MODEL_PATH)
    return model, float(val_loss)


def train_tune(
    config: Dict,
    train_ds: Subset,
    val_ds: Subset,
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_pad_idx: int,
    tgt_pad_idx: int,
) -> None:
    validate_config(config)
    device = get_device()
    model = build_model_from_config(config, src_vocab_size, tgt_vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]))
    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
        batch_size=int(config["batch_size"]),
        num_workers=0,
    )

    for epoch in range(int(config["max_epochs"])):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, src_pad_idx, tgt_pad_idx)
        val_loss = evaluate_loss(model, val_loader, criterion, device, src_pad_idx, tgt_pad_idx)
        tune.report({"loss": val_loss, "train_loss": train_loss, "epoch": epoch + 1})


def get_run_config(name: str) -> Optional[RunConfig]:
    if not RAY_AVAILABLE:
        return None

    try:
        return RunConfig(name=name, storage_path=str(RUNS_DIR.resolve()))
    except TypeError:
        return RunConfig(name=name, local_dir=str(RUNS_DIR.resolve()))


def run_hpo(
    train_ds: Subset,
    val_ds: Subset,
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_pad_idx: int,
    tgt_pad_idx: int,
    num_samples: int = 20,
    max_epochs: int = TRIAL_MAX_EPOCHS,
):
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune/Optuna is not available. Install ray[tune] and optuna.")

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "num_heads": tune.choice([4, 8]),
        "d_ff": tune.choice([1024, 2048]),
        "dropout": tune.uniform(0.1, 0.4),
        "max_epochs": max_epochs,
    }

    optuna_search = OptunaSearch(metric="loss", mode="min")
    asha = ASHAScheduler(metric="loss", mode="min", max_t=max_epochs, grace_period=2, reduction_factor=2)

    trainable = tune.with_resources(
        tune.with_parameters(
            train_tune,
            train_ds=train_ds,
            val_ds=val_ds,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_pad_idx=src_pad_idx,
            tgt_pad_idx=tgt_pad_idx,
        ),
        resources={"cpu": 8, "gpu": 1},
    )


    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=asha,
            num_samples=num_samples,
            max_concurrent_trials=1
        ),
        run_config=get_run_config("en_hi_optuna_asha"),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="loss", mode="min")
    best_config = dict(best_result.config)
    best_loss = float(best_result.metrics.get("loss", float("inf")))

    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print("Best config:", best_config)
    print(f"Best validation loss: {best_loss:.4f}")
    return results


def save_vocabs(en_vocab: Vocabulary, hi_vocab: Vocabulary) -> None:
    with open(EN_VOCAB_PATH, "wb") as f:
        pickle.dump(en_vocab, f)
    with open(HI_VOCAB_PATH, "wb") as f:
        pickle.dump(hi_vocab, f)


def write_hpo_artifacts(
    baseline_bleu: Optional[float],
    tuned_bleu: Optional[float],
    baseline_val_loss: Optional[float],
    tuned_val_loss: Optional[float],
    best_config: Optional[Dict],
) -> None:
    summary = {
        "baseline_bleu": baseline_bleu,
        "tuned_bleu_10_epochs": tuned_bleu,
        "baseline_val_loss": baseline_val_loss,
        "tuned_val_loss_10_epochs": tuned_val_loss,
        "same_or_better_bleu": bool(tuned_bleu is not None and baseline_bleu is not None and tuned_bleu >= baseline_bleu),
        "best_config": best_config,
    }

    with open(HPO_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# HPO Summary",
        "",
        f"- Baseline BLEU: {baseline_bleu if baseline_bleu is not None else 'N/A'}",
        f"- Tuned BLEU (10 epochs): {tuned_bleu if tuned_bleu is not None else 'N/A'}",
        f"- Baseline validation loss: {baseline_val_loss if baseline_val_loss is not None else 'N/A'}",
        f"- Tuned validation loss (10 epochs): {tuned_val_loss if tuned_val_loss is not None else 'N/A'}",
        f"- Same or better BLEU vs baseline: {summary['same_or_better_bleu']}",
        "",
        "## Best Config",
        "```json",
        json.dumps(best_config or {}, indent=2, ensure_ascii=False),
        "```",
    ]

    with open(HPO_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="English-Hindi Transformer training + Ray Tune HPO")
    parser.add_argument("--mode", choices=["baseline", "tune", "best", "all", "smoke"], default="all")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--trial-max-epochs", type=int, default=TRIAL_MAX_EPOCHS)
    parser.add_argument("--baseline-epochs", type=int, default=BASELINE_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_runs_dir()

    df = load_and_clean_data(args.data_path)

    en_vocab = Vocabulary(freq_threshold=2)
    hi_vocab = Vocabulary(freq_threshold=2)
    en_vocab.build_vocab(df["en"].tolist())
    hi_vocab.build_vocab(df["hi"].tolist())

    full_dataset, train_ds, val_ds = build_datasets(df, en_vocab, hi_vocab, max_len=MAX_LEN, val_ratio=0.2, seed=args.seed)
    _ = full_dataset

    src_pad_idx = en_vocab["<pad>"]
    tgt_pad_idx = hi_vocab["<pad>"]
    src_vocab_size = len(en_vocab)
    tgt_vocab_size = len(hi_vocab)
    device = get_device()

    save_vocabs(en_vocab, hi_vocab)

    baseline_model = None
    baseline_bleu = None
    baseline_val_loss = None
    tuned_bleu = None
    tuned_val_loss = None
    best_config = None

    if args.mode in {"baseline", "all", "smoke"}:
        baseline_epochs = 1 if args.mode == "smoke" else args.baseline_epochs
        baseline_model, baseline_val_loss = train_baseline(
            train_ds,
            val_ds,
            src_vocab_size,
            tgt_vocab_size,
            src_pad_idx,
            tgt_pad_idx,
            device,
            epochs=baseline_epochs,
            batch_size=60,
            resume=(args.mode != "smoke"),
        )
        baseline_bleu = evaluate_bleu_nltk(
            baseline_model,
            VAL_DATASET,
            en_vocab,
            hi_vocab,
            device,
            src_pad_idx,
            tgt_pad_idx,
            max_len=MAX_LEN,
        )
        print(f"Baseline BLEU: {baseline_bleu * 100:.2f}")

    if args.mode in {"tune", "all", "smoke"}:
        num_samples = 1 if args.mode == "smoke" else args.num_samples
        trial_epochs = 1 if args.mode == "smoke" else args.trial_max_epochs
        results = run_hpo(
            train_ds,
            val_ds,
            src_vocab_size,
            tgt_vocab_size,
            src_pad_idx,
            tgt_pad_idx,
            num_samples=num_samples,
            max_epochs=trial_epochs,
        )
        best_result = results.get_best_result(metric="loss", mode="min")
        best_config = dict(best_result.config)

    if args.mode in {"best", "all", "smoke"}:
        if best_config is None:
            if not BEST_CONFIG_PATH.exists():
                raise FileNotFoundError("No best config found. Run with --mode tune first or use --mode all.")
            with open(BEST_CONFIG_PATH, "r", encoding="utf-8") as f:
                best_config = json.load(f)
        print('Training with best config')
        best_model, tuned_val_loss = train_best_config(
            best_config,
            train_ds,
            val_ds,
            src_vocab_size,
            tgt_vocab_size,
            src_pad_idx,
            tgt_pad_idx,
            device,
            epochs=(1 if args.mode == "smoke" else args.trial_max_epochs),
        )
        tuned_bleu = evaluate_bleu_nltk(
            best_model,
            VAL_DATASET,
            en_vocab,
            hi_vocab,
            device,
            src_pad_idx,
            tgt_pad_idx,
            max_len=MAX_LEN,
        )
        print(f"Tuned BLEU ({args.trial_max_epochs} epochs): {tuned_bleu * 100:.2f}")

    if args.mode in {"all", "best", "smoke"}:
        write_hpo_artifacts(
            baseline_bleu=baseline_bleu,
            tuned_bleu=tuned_bleu,
            baseline_val_loss=baseline_val_loss,
            tuned_val_loss=tuned_val_loss,
            best_config=best_config,
        )
        print(f"Saved summary: {HPO_SUMMARY_PATH}")
        print(f"Saved report:  {HPO_REPORT_PATH}")


if __name__ == "__main__":
    main()
