# Automated Product Mapping

Maps short retail product names (`src_product_name_en`, 1–7 words) from
`product_mapping_ext_lulu.csv` to `product_code` in `product.csv`.
A code is returned only when confidence > 0.6; otherwise the row is flagged `REVIEW`.

```
 past mappings (name → code) ─┐
                              ├─► char n-gram TF-IDF index
 catalog (product.csv)  ──────┘

 src_product_name_en ─► normalize ─► top-5 nearest ─► similarity-weighted vote
                                                        │
                                       score > 0.6 ? ───┼── yes ─► AUTO   (mapped_code)
                                                        └── no  ─► REVIEW (suggestion only)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the two CSVs in data/ (git-ignored)
```

## Usage

```bash
# Validate: 5-fold out-of-fold predictions on the labelled file + accuracy summary
python map_products.py validate --out results.csv

# Predict new items (CSV with src_product_code, src_product_name_en)
python map_products.py predict --input new_items.csv --out results.csv
```

Output columns: `suggested_code`, `suggested_name`, `confidence`, `status` (AUTO/REVIEW),
`mapped_code` (filled only for AUTO). `validate` also adds `actual_code`, `is_correct`.

## Current validation (632 rows, 5-fold CV)

| Metric | Value |
|---|---|
| Top-1 accuracy (all rows) | 87.5% |
| AUTO (confidence > 0.6) | 69.0% of rows |
| AUTO precision | 96.8% |
| REVIEW | 31.0% of rows |
