"""Map short retail product names to Hassad product_code.

Approach: k-nearest-neighbours over character n-gram TF-IDF vectors.
The index holds past mappings (name -> code) plus one entry per catalog row,
so codes that were never mapped before can still be predicted.
Confidence is the similarity-weighted vote share of the winning code.

Every row gets a product_code (the best match). A row is AUTO only when
confidence > THRESHOLD, the best match is similar enough (MIN_SIMILARITY), and
the name has no frozen/dried/processed cue that the prediction contradicts;
otherwise it is REVIEW and `note` says why. Names already mapped before reuse
their existing code. The Arabic name is matched separately as a second opinion
(`arabic_check`); a disagreement does not block AUTO but marks rows to spot-check.
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
MIN_SIMILARITY = 0.5

# Words that carry no product meaning: origins, packaging, marketing, size, cut, colour.
NOISE = set("""
organic org bio eco premium fresh farm farms farmfresh local import imp imported
pp pkt pck pak pack packet packed pre tray box bag bulk loose punnet pot bunch clamshell
per approx appx apx kg g gm gms gr grm k pc pcs x s w a by with air ec pa ag sf r of and the
small medium large big jumbo mini
whole peeled sliced chopped cut cutting diced cleaned washed iced cube shredded seedless sleeve
red yellow white black golden gold pink brown purple
qatar qat doha lebanon lebnon leb usa us india iran holland holnd jordan spain thailand thai egypt
australia aus morocco moroccan oman south africa ksa saudi arabia china chinese turkey syria pakistan kenya
italy philippines vietnam chile uganda tunisia peru europe mexico azerbaijan bangladesh yemen yeman
uae sudan greek greece france netherlands new zealand nz argentina sri lanka srilanka
ethiopia colombia ecuador japan korea indonesia malaysia germany belgium cyprus portugal poland
""".split())
SYNONYMS = {"dry": "dried", "frz": "frozen", "frzn": "frozen"}

# The last two digits of product_code encode the state: 01 Fresh, 02 Frozen, 03 Dried
FRESH = "01"
STATE_CUES = {"frozen": "02", "dried": "03"}
# Processed forms have no sibling code, so a Fresh prediction for them goes to REVIEW
PROCESSED_CUES = {"salted", "roasted", "juice", "preserved", "pickled", "crystallized", "sweetened", "choco", "powder"}


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
    s = re.sub(r"\d+(\.\d+)?\s*(kgs?|gms?|grm?s?|g|ml|l|pcs|pc|k)?\b", " ", str(name).lower())
    words = [SYNONYMS.get(singular(w), singular(w)) for w in re.sub(r"[^a-z ]", " ", s).split() if len(w) > 1]
    if drop_noise:
        # Keep the original words if everything was noise
        words = [w for w in words if w not in NOISE] or words
    return " ".join(words)


def normalize_ar(name: str) -> str:
    """Strip diacritics and tatweel, unify alef/taa marbuta/yaa, keep Arabic letters only."""
    s = re.sub(r"[ً-ْـ]", "", str(name))
    s = re.sub("[أإآ]", "ا", s).replace("ة", "ه").replace("ى", "ي")
    return " ".join(re.sub(r"[^ء-ي ]", " ", s).split())


def catalog_entries(catalog: pd.DataFrame) -> tuple[list[str], list[str]]:
    """One searchable text per catalog code, e.g. 'jarjeer jarjeer roca rocca rocket'."""
    # "Fresh" is the default state, so only non-fresh states (Frozen, Dried, ...) are added
    state = catalog.state_en.where(catalog.state_en != "Fresh", "")
    text = catalog.product_en + " " + state + " " + catalog.remarks.fillna("")
    # Catalog names are clean; noise removal would break e.g. "Turkey meat", "Black tea"
    return [normalize(t, drop_noise=False) for t in text], list(catalog.product_code)


def name_key(name: str) -> str:
    """Case- and whitespace-insensitive key for exact-name lookup."""
    return " ".join(str(name).upper().split())


class KnnIndex:
    """Character n-gram TF-IDF index with a similarity-weighted top-K vote."""

    def __init__(self, docs: list[str], codes: list[str], ngram_range: tuple[int, int]):
        self.codes = np.array(codes)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True)
        self.index = self.vectorizer.fit_transform(docs)

    def vote(self, queries: list[str]) -> list[tuple[str, float, float]]:
        """(best code, vote-share confidence, top-1 similarity) per query."""
        sims = (self.vectorizer.transform(queries) @ self.index.T).toarray()
        results = []
        for row in sims:
            top = row.argsort()[::-1][:K]
            votes: dict[str, float] = {}
            for j in top:
                votes[self.codes[j]] = votes.get(self.codes[j], 0.0) + row[j]
            best = max(votes, key=votes.get)
            total = sum(votes.values())
            results.append((best, votes[best] / total if total > 0 else 0.0, row[top[0]]))
        return results


class Mapper:
    def fit(self, names, codes, catalog: pd.DataFrame, names_ar=None) -> "Mapper":
        names, codes = list(names), list(codes)
        # Names mapped before reuse their (most common) existing code
        history = pd.Series(codes, index=[name_key(n) for n in names])
        self.lookup = history.groupby(level=0).agg(lambda c: c.mode()[0]).to_dict()
        self.catalog_codes = set(catalog.product_code)

        cat_docs, cat_codes = catalog_entries(catalog)
        self.en = KnnIndex([normalize(n) for n in names] + cat_docs, codes + cat_codes, (3, 5))
        self.ar = None
        if names_ar is not None:
            ar_docs = [normalize_ar(n) for n in names_ar] + [normalize_ar(n) for n in catalog.product_ar.fillna("")]
            self.ar = KnnIndex(ar_docs, codes + list(catalog.product_code), (2, 4))
        return self

    def predict(self, names, names_ar=None) -> pd.DataFrame:
        """One row per name: product_code, confidence, similarity, auto, note, arabic_code, arabic_check."""
        names = list(names)
        normalized = [normalize(n) for n in names]
        rows = [self._predict_one(n, set(w.split()), *v) for n, w, v in zip(names, normalized, self.en.vote(normalized))]
        pred = pd.DataFrame(rows, columns=["product_code", "confidence", "similarity", "auto", "note"])

        pred["arabic_code"], pred["arabic_check"] = "", "no arabic name"
        if self.ar is not None and names_ar is not None:
            queries = [normalize_ar(n) for n in names_ar]
            ar = pd.DataFrame(self.ar.vote(queries), columns=["code", "confidence", "similarity"])
            has_ar = pd.Series([q != "" for q in queries])
            strong = (ar.confidence > THRESHOLD) & (ar.similarity >= MIN_SIMILARITY)
            # Compare at product level (first 8 digits), ignoring the state suffix
            same = ar.code.str[:8] == pred.product_code.str[:8]
            pred["arabic_code"] = ar.code.where(has_ar, "")
            pred["arabic_check"] = np.select([~has_ar, same, strong], ["no arabic name", "agree", "disagree"], "weak match")
        return pred

    def _predict_one(self, name: str, words: set, best: str, confidence: float, similarity: float) -> tuple:
        if name_key(name) in self.lookup:
            return self.lookup[name_key(name)], 1.0, 1.0, True, "exact match"

        reasons = []
        if confidence <= THRESHOLD:
            reasons.append("low confidence")
        if similarity < MIN_SIMILARITY:
            reasons.append("low similarity")
        if best.endswith(FRESH):
            state = next((STATE_CUES[w] for w in words if w in STATE_CUES), None)
            if state and best[:-2] + state in self.catalog_codes:
                best = best[:-2] + state
                note = "state from name"
                return best, confidence, similarity, not reasons, "; ".join(reasons + [note])
            if state or words & PROCESSED_CUES:
                reasons.append("name is not fresh")
        return best, confidence, similarity, not reasons, "; ".join(reasons)


def to_result(src: pd.DataFrame, pred: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Input columns with product_code/remarks filled in, plus status and review reasons."""
    names = catalog.set_index("product_code").pipe(lambda c: c.product_en + " - " + c.state_en)
    out = src.reset_index(drop=True).copy()
    out["product_code"] = pred.product_code
    out["remarks"] = out.product_code.map(names)
    out["status"] = np.where(pred.auto, "AUTO", "REVIEW")
    out["note"] = pred.note
    out["confidence"] = pred.confidence.round(3)
    out["similarity"] = pred.similarity.round(3)
    out["arabic_check"] = pred.arabic_check
    out["arabic_suggestion"] = pred.arabic_code.map(names).where(pred.arabic_check == "disagree", "")
    return out


def summarize(out: pd.DataFrame) -> None:
    """Per-source AUTO share and REVIEW reasons."""
    by = out.groupby("source")
    summary = pd.DataFrame({"rows": by.size(), "auto_share": (out.status == "AUTO").groupby(out.source).mean()})
    summary.loc["ALL"] = [len(out), (out.status == "AUTO").mean()]
    print(summary.to_string(formatters={"rows": "{:.0f}".format, "auto_share": "{:.1%}".format}))
    reasons = out[out.status == "REVIEW"].note.str.split("; ").explode().value_counts()
    print("\nREVIEW reasons (a row can have several):\n" + reasons.to_string())


def validate(mapping: pd.DataFrame, catalog: pd.DataFrame, folds: int) -> pd.DataFrame:
    """Out-of-fold predictions: every row is predicted by a model that never saw it."""
    parts = []
    for train, test in KFold(folds, shuffle=True, random_state=0).split(mapping):
        tr, te = mapping.iloc[train], mapping.iloc[test]
        model = Mapper().fit(tr.src_product_name_en, tr.product_code, catalog, tr.src_product_name_ar.fillna(""))
        parts.append(model.predict(te.src_product_name_en, te.src_product_name_ar.fillna("")).set_index(test))
    pred = pd.concat(parts).sort_index()

    actual = mapping.product_code.reset_index(drop=True)
    out = to_result(mapping, pred, catalog)
    out.insert(out.columns.get_loc("product_code") + 1, "actual_code", actual)
    out["is_correct"] = out.product_code == out.actual_code

    auto = out.status == "AUTO"
    summary = pd.DataFrame({
        "rows": out.groupby("source").size(),
        "accuracy": out.groupby("source").is_correct.mean(),
        "auto_share": auto.groupby(out.source).mean(),
        "auto_precision": out[auto].groupby("source").is_correct.mean(),
    })
    summary.loc["ALL"] = [len(out), out.is_correct.mean(), auto.mean(), out[auto].is_correct.mean()]
    print(summary.to_string(formatters={"rows": "{:.0f}".format, **{c: "{:.1%}".format for c in summary.columns[1:]}}))
    check = out[auto].groupby("arabic_check").is_correct.agg(["size", "mean"])
    print("\nAUTO precision by arabic_check:\n" + check.to_string(formatters={"mean": "{:.1%}".format}))
    return out


def read_csvs(paths: list[str]) -> pd.DataFrame:
    """Read and stack CSVs, tagging each row with its file name in `source`."""
    frames = [pd.read_csv(p, dtype=str, encoding="utf-8-sig").assign(source=Path(p).stem) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["src_product_name_ar"] = df.get("src_product_name_ar", pd.Series("", index=df.index)).fillna("")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["validate", "predict"])
    parser.add_argument("--mapping", nargs="+", required=True, help="one or more labelled mapping CSVs")
    parser.add_argument("--catalog", required=True, help="target product catalog CSV (product.csv)")
    parser.add_argument("--input", nargs="+", help="predict: one or more CSVs with src_product_name_en")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    mapping = read_csvs(args.mapping)
    catalog = pd.read_csv(args.catalog, dtype=str, encoding="utf-8-sig")

    if args.command == "validate":
        out = validate(mapping, catalog, args.folds)
    else:
        if not args.input:
            parser.error("predict requires --input")
        new = read_csvs(args.input)
        model = Mapper().fit(mapping.src_product_name_en, mapping.product_code, catalog, mapping.src_product_name_ar)
        out = to_result(new, model.predict(new.src_product_name_en, new.src_product_name_ar), catalog)
        summarize(out)

    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
