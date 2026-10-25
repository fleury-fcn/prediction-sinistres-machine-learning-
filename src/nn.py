"""Tabular MLP with categorical embeddings (PyTorch), wrapped as an sklearn estimator.

The estimator receives the feature table produced by
:class:`src.features.ClaimFeatureBuilder` (numeric floats + string
categoricals). All preprocessing (median imputation, standardisation,
category -> index maps) is fitted inside :meth:`fit`, on the training data it
receives, so it is fold-safe when used in cross-validation.

Early stopping uses an internal stratified validation split carved out of the
training data passed to ``fit`` - never the evaluation fold.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn


class _MLP(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        n_numeric: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, min(16, (card + 1) // 2)) for card in cardinalities]
        )
        in_dim = n_numeric + sum(e.embedding_dim for e in self.embeddings)
        layers: list[nn.Module] = []
        for width in hidden_dims:
            layers += [nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.ReLU()]
            layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.body = nn.Sequential(*layers)
        self.embedding_dropout = nn.Dropout(dropout / 2)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if parts:
            x = torch.cat([self.embedding_dropout(torch.cat(parts, dim=1)), x_num], dim=1)
        else:
            x = x_num
        return self.body(x).squeeze(1)


class TabularMLPClassifier(ClassifierMixin, BaseEstimator):
    """MLP with entity embeddings, batch norm, dropout and early stopping."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        val_fraction: float = 0.15,
        class_weight: str | None = None,
        random_state: int = 42,
        n_threads: int = 1,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_threads = n_threads

    # -- preprocessing (fitted on training data only) -------------------------
    def _fit_encoders(self, X: pd.DataFrame) -> None:
        self.cat_cols_ = [c for c in X.columns if X[c].dtype == object]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        # index 0 is reserved for unseen / missing levels
        self.cat_maps_ = {
            c: {v: i + 1 for i, v in enumerate(sorted(X[c].astype(str).unique()))}
            for c in self.cat_cols_
        }
        num = X[self.num_cols_].astype(float)
        self.num_median_ = num.median()
        filled = num.fillna(self.num_median_)
        self.num_mean_ = filled.mean()
        self.num_std_ = filled.std(ddof=0).replace(0, 1.0)

    def _encode(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        cats = (
            np.stack(
                [
                    X[c].astype(str).map(self.cat_maps_[c]).fillna(0).to_numpy()
                    for c in self.cat_cols_
                ],
                axis=1,
            )
            if self.cat_cols_
            else np.zeros((len(X), 0))
        )
        num = X[self.num_cols_].astype(float).fillna(self.num_median_)
        num = ((num - self.num_mean_) / self.num_std_).to_numpy()
        return (
            torch.as_tensor(cats, dtype=torch.long),
            torch.as_tensor(num, dtype=torch.float32),
        )

    # -- training --------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> TabularMLPClassifier:
        torch.manual_seed(self.random_state)
        torch.set_num_threads(self.n_threads)  # see predict_proba
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])

        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=self.val_fraction, stratify=y, random_state=self.random_state
        )
        self._fit_encoders(X_fit)
        cat_fit, num_fit = self._encode(X_fit)
        cat_val, num_val = self._encode(X_val)
        y_fit_t = torch.as_tensor(y_fit)
        y_val_t = torch.as_tensor(y_val)

        cards = [len(self.cat_maps_[c]) + 1 for c in self.cat_cols_]
        self.model_ = _MLP(cards, len(self.num_cols_), tuple(self.hidden_dims), self.dropout)
        pos_weight = None
        if self.class_weight == "balanced":
            pos_weight = torch.tensor((1 - y_fit.mean()) / max(y_fit.mean(), 1e-6))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        val_loss_fn = nn.BCEWithLogitsLoss()  # unweighted, comparable across settings
        optimiser = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        n = len(y_fit)
        best_loss, best_state, stale = np.inf, None, 0
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_auc": []}
        for _epoch in range(self.max_epochs):
            self.model_.train()
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs > 1 sample
                    continue
                optimiser.zero_grad()
                loss = loss_fn(self.model_(cat_fit[idx], num_fit[idx]), y_fit_t[idx])
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(idx)
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(cat_val, num_val)
                val_loss = val_loss_fn(val_logits, y_val_t).item()
            self.history_["train_loss"].append(epoch_loss / n)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_auc"].append(
                float(roc_auc_score(y_val, torch.sigmoid(val_logits).numpy()))
            )
            if val_loss < best_loss - 1e-4:
                best_loss, best_state, stale = val_loss, copy.deepcopy(self.model_.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin(self.history_["val_loss"])) + 1
        self.model_.eval()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Torch's OpenMP pool deadlocks on macOS once LightGBM/XGBoost have used
        # theirs; a single thread is plenty for this network size.
        torch.set_num_threads(self.n_threads)
        cat, num = self._encode(X)
        with torch.no_grad():
            p = torch.sigmoid(self.model_(cat, num)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
