"""Linha de base clássica: TF-IDF + um classificador de 4 classes por aspecto."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..config import ASPECTS, LABEL_COLUMNS, POLARITIES, SEED

ESTIMATORS = {
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0),
    "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=SEED),
}


class TfidfAspectClassifier:
    """Um vetorizador compartilhado e um classificador independente por aspecto.

    Os textos devem chegar já limpos com `preprocess.clean_classic`.
    """

    def __init__(self, estimator="logreg", ngram_range=(1, 2), min_df=2, max_features=30000):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df,
                                          max_features=max_features, sublinear_tf=True)
        self.base = ESTIMATORS[estimator] if isinstance(estimator, str) else estimator
        self.models = {}

    def fit(self, texts, labels: pd.DataFrame):
        X = self.vectorizer.fit_transform(texts)
        for asp, col in zip(ASPECTS, LABEL_COLUMNS):
            y = labels[col].to_numpy()
            if len(set(y)) == 1:  # aspecto sem variação no treino: prevê a constante
                self.models[asp] = y[0]
            else:
                self.models[asp] = clone(self.base).fit(X, y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        """Array (n_textos, n_aspectos, 4) na ordem de POLARITIES."""
        X = self.vectorizer.transform(texts)
        out = np.zeros((X.shape[0], len(ASPECTS), len(POLARITIES)))
        for j, asp in enumerate(ASPECTS):
            m = self.models[asp]
            if isinstance(m, str):
                out[:, j, POLARITIES.index(m)] = 1.0
                continue
            proba = m.predict_proba(X)
            for k, cls in enumerate(m.classes_):
                out[:, j, POLARITIES.index(cls)] = proba[:, k]
        return out

    def predict(self, texts) -> pd.DataFrame:
        return proba_to_labels(self.predict_proba(texts))

    def top_terms(self, aspecto, polaridade, k=15):
        """Termos de maior peso (só para logreg) — explicabilidade global simples."""
        m = self.models[aspecto]
        idx = list(m.classes_).index(polaridade)
        coefs = m.coef_[idx] if m.coef_.shape[0] > 1 else m.coef_[0]
        vocab = np.array(self.vectorizer.get_feature_names_out())
        return list(vocab[np.argsort(coefs)[::-1][:k]])


def proba_to_labels(proba: np.ndarray) -> pd.DataFrame:
    idx = proba.argmax(axis=2)
    return pd.DataFrame({col: [POLARITIES[i] for i in idx[:, j]] for j, col in enumerate(LABEL_COLUMNS)})
