# Automated Product Mapping

Maps short retail product names (`src_product_name_en`) to Hassad `product_code` (`product.csv`),
learning from retailers' past mappings. Every row gets a `product_code`; rows the model is not sure
about are marked `REVIEW` with the reason.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Map new products (one or more input CSVs -> one output CSV)
python map_products.py predict \
  --mapping product_mapping_ext_lulu.csv product_mapping_ext_megamart.csv product_mapping_ext_meera.csv \
  --catalog product.csv \
  --input   lulu_unmapped.csv meera_unmapped.csv \
  --out     results.csv

# Optional: measure accuracy on the labelled files (5-fold cross-validation)
python map_products.py validate \
  --mapping product_mapping_ext_lulu.csv product_mapping_ext_megamart.csv product_mapping_ext_meera.csv \
  --catalog product.csv --out validation.csv
```

Inputs need `src_product_name_en`; `src_product_name_ar` is optional (enables the Arabic check).
CSV files are never committed; always pass their paths.

## Reading the output

The output keeps all input columns and fills `product_code` and `remarks`, plus:

| Column | Meaning |
|---|---|
| `source` | Input file the row came from |
| `status` | `AUTO` = accept as is · `REVIEW` = a person should confirm `product_code` |
| `note` | Why: `exact match`, `state from name`, `low confidence`, `low similarity`, `name is not fresh` |
| `confidence` / `similarity` | Vote share of the top-5 matches / closeness of the best match (0–1) |
| `arabic_check` | Arabic name matched separately: `agree`, `disagree`, `weak match`, `no arabic name` |
| `arabic_suggestion` | The Arabic match's product when it disagrees |

Suggested workflow: work through `REVIEW` rows, then spot-check `AUTO` rows with
`arabic_check = disagree`. Add confirmed rows back to the mapping files so the model learns them.

## How it works

```
 past mappings (1..n CSVs) ─┐
                            ├─► char n-gram TF-IDF index ─► top-5 nearest ─► weighted vote
 product.csv catalog ───────┘
                                     exact name seen before? ─► reuse its code
 AUTO only if: confidence > 0.6 AND similarity ≥ 0.5 AND name is not frozen/dried/processed
               while the prediction is Fresh (a "frozen"/"dried" name switches to the matching code)
```

## Current results

| Data | AUTO share | AUTO precision |
|---|---|---|
| Labelled files, 5-fold CV (4,186 rows) | 89.8% | 97.4% (measured) |
| New unmapped files (6,362 rows, 4 retailers) | 72.9% | ~91% (hand-checked sample of 198) |
