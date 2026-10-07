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
| `size_value` / `size_unit` | Size parsed from the English name, e.g. `400.0` / `g`; blank when the name has none |
| `size_multiplier` | Pack count when present: `3x400g`, `400g 3s`, `400gX5`; blank otherwise |

Suggested workflow: work through `REVIEW` rows, then spot-check `AUTO` rows with
`arabic_check = disagree`. Add confirmed rows back to the mapping files so the model learns them.

## Size extraction

Regex only, from `src_product_name_en`. Units are canonical lowercase, values are not converted:
`g` (g, gm, gms, gr, grm, gram) · `kg` (kg, kgs, kilo, 1.2k) · `ml` · `l` (l, ltr, litre) · `oz` · `lb` · `pint` ·
`pcs` (pc, pcs, piece, pkt, pack, pk, cob; only when there is no weight/volume, and at most 100).

| Name | size_value | size_unit | size_multiplier |
|---|---|---|---|
| `DriscollsBlueberryPortugal125g` | 125.0 | g | |
| `Lulu Molokhia 400g 3s P/O` | 400.0 | g | 3.0 |
| `Mccain Onion Rings 2x400g PO` | 400.0 | g | 2.0 |
| `CAPSICUM MIX 3PCS` | 3.0 | pcs | |
| `NECTARINE 1X6 IRAN` | 6.0 | pcs | 1.0 |
| `CAPSICUMS RED 70/90 FL`, `MANGO R2E2`, `Fries 6mm`, `GRADE 1`, `100%` | | | |

Grades, calibres and ranges (`70/90`, `14-16`, `70S`), cut sizes (`6mm`), varieties (`R2E2`) and
ambiguous unitless multipacks (`4X250`) are left blank. Test cases: `pip install pytest && python -m pytest tests`.

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
