"""Map short retail product names to Hassad product_code.

Approach: k-nearest-neighbours over character n-gram TF-IDF vectors.
The index holds past mappings (name -> code) plus one entry per catalog row,
so codes that were never mapped before can still be predicted.
Confidence is the similarity-weighted vote share of the winning code.
"""

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

K = 5
THRESHOLD = 0.6


def normalize(name: str) -> str:
    """Lowercase, drop pack sizes (500gm, 1.5kg), punctuation and 1-char tokens."""
    s = name.lower()
    s = re.sub(r"\d+(\.\d+)?\s*(kg|gm|g|ml|l|pcs|pc)?\b", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(w for w in s.split() if len(w) > 1)


def catalog_entries(catalog: pd.DataFrame) -> tuple[list[str], list[str]]:
    """One searchable text per catalog code, e.g. 'jarjeer fresh jarjeer roca rocca rocket'."""
    text = catalog.product_en + " " + catalog.state_en + " " + catalog.remarks.fillna("")
    return [normalize(t) for t in text], list(catalog.product_code)


class Mapper:
    def fit(self, names, codes, catalog: pd.DataFrame) -> "Mapper":
        cat_docs, cat_codes = catalog_entries(catalog)
        docs = [normalize(n) for n in names] + cat_docs
        self.codes = np.array(list(codes) + cat_codes)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.index = self.vectorizer.fit_transform(docs)
        return self

    def predict(self, names) -> tuple[list[str], list[float]]:
        sims = (self.vectorizer.transform([normalize(n) for n in names]) @ self.index.T).toarray()
        codes, scores = [], []
        for row in sims:
            votes: dict[str, float] = {}
            for j in row.argsort()[::-1][:K]:
                votes[self.codes[j]] = votes.get(self.codes[j], 0.0) + row[j]
            best = max(votes, key=votes.get)
            total = sum(votes.values())
            codes.append(best)
            scores.append(votes[best] / total if total > 0 else 0.0)
        return codes, scores


def to_result(src: pd.DataFrame, codes, scores, catalog: pd.DataFrame) -> pd.DataFrame:
    names = catalog.set_index("product_code").pipe(lambda c: c.product_en + " - " + c.state_en)
    out = src[["src_product_code", "src_product_name_en"]].copy()
    out["suggested_code"] = codes
    out["suggested_name"] = out.suggested_code.map(names)
    out["confidence"] = np.round(scores, 3)
    out["status"] = np.where(out.confidence > THRESHOLD, "AUTO", "REVIEW")
    out["mapped_code"] = out.suggested_code.where(out.status == "AUTO", "")
    return out


def validate(mapping: pd.DataFrame, catalog: pd.DataFrame, folds: int) -> pd.DataFrame:
    """Out-of-fold predictions: every row is predicted by a model that never saw it."""
    codes = np.empty(len(mapping), dtype=object)
    scores = np.zeros(len(mapping))
    for train, test in KFold(folds, shuffle=True, random_state=0).split(mapping):
        model = Mapper().fit(mapping.src_product_name_en.iloc[train], mapping.product_code.iloc[train], catalog)
        codes[test], scores[test] = model.predict(mapping.src_product_name_en.iloc[test])

    out = to_result(mapping, codes, scores, catalog)
    out["actual_code"] = mapping.product_code
    out["is_correct"] = out.suggested_code == out.actual_code

    auto = out[out.status == "AUTO"]
    print(f"Rows                 : {len(out)}")
    print(f"Top-1 accuracy (all) : {out.is_correct.mean():.1%}")
    print(f"AUTO (conf > {THRESHOLD})   : {len(auto)} ({len(auto) / len(out):.1%})  precision {auto.is_correct.mean():.1%}")
    print(f"REVIEW               : {len(out) - len(auto)} ({1 - len(auto) / len(out):.1%})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "predict"])
    parser.add_argument("--mapping", default="data/product_mapping_ext_lulu.csv", help="labelled past mappings")
    parser.add_argument("--catalog", default="data/product.csv", help="target product catalog")
    parser.add_argument("--input", help="predict: CSV with src_product_code, src_product_name_en")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    mapping = pd.read_csv(args.mapping, dtype=str)
    catalog = pd.read_csv(args.catalog, dtype=str, encoding="utf-8-sig")

    if args.command == "validate":
        out = validate(mapping, catalog, args.folds)
    else:
        if not args.input:
            parser.error("predict requires --input")
        new = pd.read_csv(args.input, dtype=str)
        model = Mapper().fit(mapping.src_product_name_en, mapping.product_code, catalog)
        out = to_result(new, *model.predict(new.src_product_name_en), catalog)
        print(out.status.value_counts().to_string())

    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
