"""Encoder Transformer (BERTimbau, Albertina) com uma cabeça softmax por aspecto.

Arquitetura: texto -> encoder -> vetor [CLS] -> dropout -> linear com
n_aspectos x 4 saídas -> uma softmax de 4 classes por aspecto. A perda é a
soma das entropias cruzadas dos aspectos, com pesos por classe opcionais
para o desbalanceamento.
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from ..config import ASPECTS, LABEL_COLUMNS, POL2ID, POLARITIES, SEED
from ..metrics import PRIMARY_METRIC
from .baseline import proba_to_labels

ENCODERS = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "albertina": "PORTULAN/albertina-100m-portuguese-ptbr-encoder",
}


@dataclass
class TrainConfig:
    model_name: str = ENCODERS["bertimbau"]
    max_len: int = 128
    batch_size: int = 32
    lr: float = 3e-5
    epochs: int = 4
    warmup: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    class_weights: bool = True
    fp16: bool = True
    seed: int = SEED


class MultiAspectModel(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(ASPECTS) * len(POLARITIES))

    def forward(self, input_ids, attention_mask, **kwargs):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(self.dropout(cls)).view(-1, len(ASPECTS), len(POLARITIES))


class _TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        if labels is None:
            self.labels = None
        else:
            self.labels = labels[LABEL_COLUMNS].apply(lambda s: s.map(POL2ID)).to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return i


def _class_weights(labels: pd.DataFrame) -> torch.Tensor:
    """Peso inverso à frequência (suavizado por raiz) para cada classe de cada aspecto."""
    w = np.ones((len(ASPECTS), len(POLARITIES)), dtype=np.float32)
    for j, col in enumerate(LABEL_COLUMNS):
        counts = labels[col].value_counts().reindex(POLARITIES, fill_value=0).to_numpy() + 1
        w[j] = np.sqrt(counts.sum() / (len(POLARITIES) * counts))
    return torch.tensor(w)


class EncoderAspectClassifier:
    def __init__(self, config: TrainConfig | None = None, device=None):
        self.cfg = config or TrainConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        torch.manual_seed(self.cfg.seed)
        self.model = MultiAspectModel(self.cfg.model_name, self.cfg.dropout).to(self.device)
        self.history = []

    def _loader(self, texts, labels=None, shuffle=False):
        ds = _TextDataset(texts, labels)

        def collate(idx):
            enc = self.tokenizer([ds.texts[i] for i in idx], truncation=True, max_length=self.cfg.max_len,
                                 padding=True, return_tensors="pt")
            if ds.labels is not None:
                enc["labels"] = torch.from_numpy(ds.labels[idx])
            return enc

        g = torch.Generator().manual_seed(self.cfg.seed)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle, collate_fn=collate, generator=g)

    def fit(self, texts, labels: pd.DataFrame, dev_texts=None, dev_labels=None, save_path=None):
        cfg = self.cfg
        loader = self._loader(texts, labels, shuffle=True)
        weights = _class_weights(labels).to(self.device) if cfg.class_weights else None
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total = len(loader) * cfg.epochs
        sched = get_linear_schedule_with_warmup(opt, math.ceil(cfg.warmup * total), total)
        use_amp = cfg.fp16 and self.device == "cuda"  # T4 não tem bf16: usar fp16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best = -1.0

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            t0, running = time.time(), 0.0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = self.model(**batch)
                    loss = self._loss(logits.float(), batch["labels"], weights)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()

            log = {"epoca": epoch, "perda_treino": running / len(loader), "segundos": time.time() - t0}
            if dev_texts is not None:
                log["f1_dev"] = PRIMARY_METRIC(dev_labels.reset_index(drop=True), self.predict(dev_texts))
                if save_path and log["f1_dev"] > best:
                    best = log["f1_dev"]
                    self.save(save_path)
            self.history.append(log)
            print(log)
        return self

    @staticmethod
    def _loss(logits, labels, weights):
        loss = 0.0
        for j in range(len(ASPECTS)):
            w = None if weights is None else weights[j]
            loss = loss + nn.functional.cross_entropy(logits[:, j], labels[:, j], weight=w)
        return loss / len(ASPECTS)

    @torch.no_grad()
    def predict_proba(self, texts) -> np.ndarray:
        self.model.eval()
        out = []
        for batch in self._loader(texts):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out.append(torch.softmax(self.model(**batch).float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, texts) -> pd.DataFrame:
        return proba_to_labels(self.predict_proba(texts))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self
