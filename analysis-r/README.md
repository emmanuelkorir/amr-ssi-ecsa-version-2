# R Markdown Workflow: Deduplication + Meta-analysis for AMR/SSI (ECSA)

This R Markdown workflow provides:

- Duplicate publication detection and removal with complete logs.
- Stratified random-effects meta-analysis of proportions (SSI incidence; AMR by pathogen–antibiotic; mortality).
- Meta-regression (optional, via covariates in config).
- Pooling of risk factor effects (OR/RR/HR).
- Qualitative thematic aggregation.

## How to run in RStudio

1. Configure paths and column mappings in `analysis-r/config.yml` (set `data_dir` to your processed CSVs, e.g., `outputs/processed`).
2. Open `analysis-r/pipeline.Rmd` and click Knit (or run chunks sequentially).
3. Outputs will be written to `analysis/results_r/` with duplicate logs under `analysis/results_r/dedup/`.

## Individual notebooks

- `00_setup.Rmd`: installs packages and defines helper functions.
- `01_deduplicate.Rmd`: detects duplicates, selects a primary record per cluster, writes full logs and deduplicated CSVs.
- `02_meta_proportions.Rmd`: SSI/AMR/mortality meta-analyses with forest/funnel plots.
- `03_meta_risk_factors.Rmd`: pools OR/RR/HR.
- `04_qualitative_thematic.Rmd`: validates themes and summarizes.
- `pipeline.Rmd`: orchestrates everything as a single run.

## Dedup outputs

- `publications_registry.csv`, `duplicate_pairs_scored.csv`, `duplicate_clusters.csv`, `duplicate_decisions_map.csv`
- `<outcome>_dedup.csv`, `<outcome>_dropped.csv`
- `identical_result_flags_all.csv`, `summary.txt`

## Notes

- Proportions use `meta::metaprop` (default `PLOGIT`; switch to `PFT` in config for extreme proportions or small n).
- Meta-regression uses `metareg`; coefficients are on the transformed scale.
- AMR meta-regression across all pairs is exploratory; primary reporting should be per pathogen–antibiotic.
