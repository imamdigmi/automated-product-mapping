# Automated Product Mapping

Maps short retail product names (`src_product_name_en`, 1–8 words) from retailer mapping files
(e.g. `product_mapping_ext_lulu.csv`) to `product_code` in `product.csv`.
A code is returned only when confidence > 0.6; otherwise the row is flagged `REVIEW`.

```
 past mappings (1..n retailer CSVs) ─┐
                                     ├─► char n-gram TF-IDF index
 catalog (product.csv)  ─────────────┘

 src_product_name_en ─► normalize ─► top-5 nearest ─► similarity-weighted vote
   (drop sizes, origins,                                 │
    packaging, colours;          score > 0.6 ? ──────────┼── yes ─► AUTO   (mapped_code)
    plural → singular)                                   └── no  ─► REVIEW (suggestion only)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CSV files are never committed; pass their paths as parameters.

## Usage

```bash
# Validate: 5-fold out-of-fold predictions on the labelled files + per-file accuracy summary
python map_products.py validate \
  --mapping product_mapping_ext_lulu.csv product_mapping_ext_megamart.csv product_mapping_ext_meera.csv \
  --catalog product.csv --out results.csv

# Predict new items (CSV with src_product_code, src_product_name_en)
python map_products.py predict \
  --mapping product_mapping_ext_lulu.csv product_mapping_ext_megamart.csv product_mapping_ext_meera.csv \
  --catalog product.csv --input new_items.csv --out results.csv
```

Output columns: `source`, `suggested_code`, `suggested_name`, `confidence`, `status` (AUTO/REVIEW),
`mapped_code` (filled only for AUTO). `validate` also adds `actual_code`, `is_correct`.

## Current validation (5-fold CV, all three retailers pooled)

| File | Rows | Accuracy | AUTO share | AUTO precision |
|---|---|---|---|---|
| lulu | 632 | 87.8% | 87.0% | 94.7% |
| megamart | 1,691 | 93.5% | 92.5% | 97.1% |
| meera | 1,863 | 92.5% | 91.7% | 96.7% |
| **All** | **4,186** | **92.2%** | **91.3%** | **96.5%** |

Most remaining AUTO errors are label conflicts between retailers (e.g. plums labelled Peach,
kale as Other brassicas vs Other Leafy Greens), not model errors.
