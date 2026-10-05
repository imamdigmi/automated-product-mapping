"""Map short retail product names to Hassad product_code.

Approach: k-nearest-neighbours over character n-gram TF-IDF vectors.
The index holds past mappings (name -> code) plus one entry per catalog row,
so codes that were never mapped before can still be predicted.
Confidence is the similarity-weighted vote share of the winning code.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

K = 5
THRESHOLD = 0.6


# Words that carry no product meaning: origins, packaging, marketing, size, cut, colour.
NOISE = set("""
organic org bio eco premium fresh farm farms farmfresh local import imp imported
pp pkt pck pak pack packet packed pre tray box bag bulk loose punnet pot bunch clamshell
per approx appx apx kg g gm gms gr grm k pc pcs x s w a by with air ec pa ag sf r of and the
small medium large big jumbo mini
whole peeled sliced chopped cut cutting diced cleaned washed iced cube shredded seedless sleeve
red yellow white black golden gold pink brown purple
qatar qat doha lebanon lebnon leb usa us india iran holland holnd jordan spain thailand thai egypt
australia aus morocco moroccan oman south africa ksa saudi arabia china chinese syria pakistan kenya
italy philippines vietnam chile uganda tunisia peru europe mexico azerbaijan bangladesh yemen yeman
uae sudan greek greece france netherlands new zealand nz argentina sri lanka srilanka
ethiopia colombia ecuador japan korea indonesia malaysia germany belgium cyprus portugal poland
""".split())
SYNONYMS = {"dry": "dried", "frz": "frozen", "frzn": "frozen"}


def singular(word: str) -> str:
    """Crude plural stripping so 'cherries' matches 'cherry' and 'plums' matches 'plum'."""
    if len(word) <= 3 or word.endswith(("ss", "us")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    return word[:-1] if word.endswith("s") else word


def normalize(name: str, drop_noise: bool = True) -> str:
    """Lowercase, drop pack sizes and noise words, singularize, map state synonyms."""
    s = re.sub(r"\d+(\.\d+)?\s*(kgs?|gms?|grm?s?|g|ml|l|pcs|pc|k)?\b", " ", name.lower())
    words = [SYNONYMS.get(singular(w), singular(w)) for w in re.sub(r"[^a-z ]", " ", s).split() if len(w) > 1]
    if drop_noise:
        # Keep the original words if everything was noise
        words = [w for w in words if w not in NOISE] or words
    return " ".join(words)


def catalog_entries(catalog: pd.DataFrame) -> tuple[list[str], list[str]]:
    """One searchable text per catalog code, e.g. 'jarjeer jarjeer roca rocca rocket'."""
    # "Fresh" is the default state, so only non-fresh states (Frozen, Dried, ...) are added
    state = catalog.state_en.where(catalog.state_en != "Fresh", "")
    text = catalog.product_en + " " + state + " " + catalog.remarks.fillna("")
    # Catalog names are clean; noise removal would break e.g. "Turkey meat", "Black tea"
    return [normalize(t, drop_noise=False) for t in text], list(catalog.product_code)


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
    out = src[[c for c in ["source", "src_product_code", "src_product_name_en"] if c in src]].copy()
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

    auto = out.status == "AUTO"
    summary = pd.DataFrame({
        "rows": out.groupby("source").size(),
        "accuracy": out.groupby("source").is_correct.mean(),
        "auto_share": auto.groupby(out.source).mean(),
        "auto_precision": out[auto].groupby("source").is_correct.mean(),
    })
    summary.loc["ALL"] = [len(out), out.is_correct.mean(), auto.mean(), out[auto].is_correct.mean()]
    print(f"AUTO = confidence > {THRESHOLD}")
    print(summary.to_string(formatters={"rows": "{:.0f}".format, **{c: "{:.1%}".format for c in summary.columns[1:]}}))
    return out


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "predict"])
    parser.add_argument("--mapping", nargs="+", required=True, help="one or more labelled mapping CSVs")
    parser.add_argument("--catalog", required=True, help="target product catalog CSV (product.csv)")
    parser.add_argument("--input", help="predict: CSV with src_product_code, src_product_name_en")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    mapping = pd.concat([read_csv(p).assign(source=Path(p).stem) for p in args.mapping], ignore_index=True)
    catalog = read_csv(args.catalog)

    if args.command == "validate":
        out = validate(mapping, catalog, args.folds)
    else:
        if not args.input:
            parser.error("predict requires --input")
        new = read_csv(args.input)
        model = Mapper().fit(mapping.src_product_name_en, mapping.product_code, catalog)
        out = to_result(new, *model.predict(new.src_product_name_en), catalog)
        print(out.status.value_counts().to_string())

    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
