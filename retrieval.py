from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import read_json, read_jsonl, resolve, write_json


def evidence_text(evidence):
    return json.dumps({key: evidence[key] for key in ("A", "D", "F", "P")},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class HistoricalCaseRetriever:
    def __init__(self, encoder, alpha=.2, top_k=5):
        if not 0 <= alpha < 1 or top_k < 1:
            raise ValueError("Invalid graph diffusion settings")
        self.encoder, self.alpha, self.top_k = encoder, alpha, top_k
        self.cases, self.embeddings = [], None

    def build(self, references_jsonl, output):
        source = Path(references_jsonl)
        cases, texts, identities = [], [], set()
        for row in read_jsonl(source):

            if row.get("split") not in ("reference", "train"):
                raise ValueError("Historical index accepts reference/train cases only")
            if row["case_id"] in identities:
                raise ValueError("Duplicate reference case_id")
            identities.add(row["case_id"])
            content = read_json(resolve(source.parent, row["evidence_path"]))
            evidence = content.get("evidence", content)
            texts.append(evidence_text(evidence))
            report = resolve(source.parent, row["report_path"]).read_text(encoding="utf-8")
            cases.append({"case_id": row["case_id"], "patient_id": row["patient_id"], "diagnostic_report": report})
        if not cases:
            raise ValueError("Empty historical case library")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.cases, self.embeddings = cases, self.encoder.encode(texts)
        write_json(output/"index.json", {"cases": cases, "encoder_identity": self.encoder.identity,
                   "alpha": self.alpha, "top_k": self.top_k})
        np.savez_compressed(output/"embeddings.npz", embeddings=self.embeddings)

    def load(self, folder):
        metadata = read_json(Path(folder)/"index.json")
        if metadata["encoder_identity"] != self.encoder.identity:
            raise ValueError("Query encoder differs from historical-index encoder")
        self.cases = metadata["cases"]
        with np.load(Path(folder)/"embeddings.npz", allow_pickle=False) as data:
            self.embeddings = data["embeddings"]
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.cases) or not np.isfinite(self.embeddings).all():
            raise ValueError("Invalid historical embedding matrix")

    def retrieve(self, case, evidence):
        if self.embeddings is None:
            return []
        keep = [i for i, reference in enumerate(self.cases)
                if reference["case_id"] != case["case_id"] and reference["patient_id"] != case["patient_id"]]
        if not keep:
            return []
        z = self.embeddings[keep]
        q = np.maximum(0, z@self.encoder.encode([evidence_text(evidence)])[0])
        w = np.maximum(0, z@z.T)
        np.fill_diagonal(w, 0)
        sums = w.sum(axis=1, keepdims=True)
        w = np.divide(w, sums, out=np.zeros_like(w), where=sums > 0)

        scores = np.linalg.solve((np.eye(len(keep))-self.alpha*w).T, q)
        return [{**self.cases[keep[i]], "score": float(scores[i])} for i in np.argsort(-scores, kind="stable")[:self.top_k]]
