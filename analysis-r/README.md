# R-Based Analysis Pipeline for AMR-SSI Systematic Review

## 1. Overview

This directory, `analysis-r/`, contains a complete, reproducible, and robust R-based workflow for the systematic review: "Challenges, Innovations, and Strategies to Combat Antimicrobial Resistance in Surgical Site Infections in East, Central, and Southern Africa."

The pipeline is designed for **extreme robustness** and **full auditability**. It automates deduplication, quantitative meta-analyses, and qualitative summaries, transforming raw, processed data into publication-ready results.

### Core Principles

- **Zero-Input Resilience**: The pipeline will run to completion without errors even if the input data directory (`outputs/processed/`) is empty. It will produce empty, correctly-formatted logs and gracefully skip analyses, ensuring workflow integrity under all conditions.
- **Pervasive NA-Handling**: Every script is built to anticipate and correctly handle missing (`NA`) data. Data is never imputed. Instead, incomplete records are logged and excluded from specific calculations, ensuring analytical validity.
- **Full Auditability**: The entire deduplication process is transparent and logged. An external reviewer can trace every decision, from identifying potential duplicates to selecting the primary record. All data exclusions for any analysis are recorded in `analysis/results_r/missing_data_log.csv`.

## 2. Pipeline Structure

The workflow is orchestrated by `pipeline.Rmd`, which executes a series of modular R Markdown notebooks in sequence. All configuration is managed centrally in `config.yml`.

- `pipeline.Rmd`: The master script. Knitting this file runs the entire analysis from start to finish.
- `config.yml`: Central configuration for file paths, analysis parameters, and deduplication thresholds.
- `00_setup.Rmd`: Installs packages, loads helper functions, and reads all input data. Handles the zero-input scenario by creating placeholder data structures.
- `01_deduplicate.Rmd`: Implements a sophisticated, two-pronged deduplication algorithm (exact and fuzzy matching) and generates comprehensive audit logs.
- `02_meta_proportions.Rmd`: Performs meta-analyses and meta-regressions for proportional data (SSI incidence, mortality, AMR).
- `03_meta_risk_factors.Rmd`: Pools effect sizes (OR/RR/HR) for risk factor analyses.
- `04_qualitative_thematic.Rmd`: Summarizes and visualizes qualitative thematic data.

## 3. Execution Instructions

**Prerequisites**:

- R and RStudio installed.
- The R project is rooted in the main repository directory.

**To run the entire pipeline:**

1.  Open the R project.
2.  Open the `analysis-r/pipeline.Rmd` file in RStudio.
3.  Click the "Knit" button.

The script will execute all analysis steps and deposit the results in the `analysis/results_r/` directory.

## 4. Deduplication Logic

The `01_deduplicate.Rmd` script is the gatekeeper of the pipeline, ensuring that each unique publication is represented only once. The logic is as follows:

1.  **Master Registry**: A master list of all publications is created from the `study_metadata.csv` and other input files.
2.  **Candidate Pair Identification**:
    - **Exact Matches**: Pairs are created from records sharing the same DOI or PMID.
    - **Fuzzy Matches**: Pairs are created from records that meet all of the following criteria, defined in `config.yml`:
      - Title similarity (Levenshtein distance) is below `title_levenshtein_dist`.
      - Author list similarity (Jaccard index) is above `author_jaccard_min`.
      - Publication year difference is no more than `year_diff_max`.
3.  **Cluster Resolution**: The `igraph` package is used to resolve the list of pairs into distinct clusters of duplicate publications.
4.  **Primary Record Selection**: Within each cluster, one record is designated as the "primary" record based on a strict, predefined hierarchy set in `config.yml` (e.g., prefer records with a DOI, then PMID, then the most complete data).
5.  **Auditable Output**: The entire process is logged in the `analysis/results_r/dedup/` directory, including the master registry, all scored pairs, the final clusters, and the decision map showing which records were kept or discarded.
6.  **Application**: The final decision map is used to filter all input datasets, creating deduplicated versions for downstream analysis.

## 5. Output Manifest

All outputs are stored in `analysis/results_r/`.

- `missing_data_log.csv`: A comprehensive log of all records excluded from any analysis due to missing data.
- `dedup/`: A directory containing the full audit trail of the deduplication process.
  - `01_master_registry.csv`: List of all unique publications found.
  - `02_candidate_pairs.csv`: All potential duplicate pairs identified.
  - `03_duplicate_clusters.csv`: Final clusters of duplicate records.
  - `04_deduplication_map.csv`: The final decision map used to deduplicate datasets.
- `plots/`: Contains all generated figures (e.g., forest plots, funnel plots, bar charts).
- `meta_analysis_ssi_incidence.csv`: Results of the SSI incidence meta-analysis.
- `meta_analysis_mortality.csv`: Results of the mortality meta-analysis.
- `meta_analysis_amr_proportions.csv`: Results of the AMR proportions meta-analyses.
- `meta_regression_results.csv`: Results of the temporal trend meta-regressions.
- `pooled_risk_factors.csv`: Pooled effect estimates for risk factors.
- `qualitative_theme_summary.csv`: Frequency counts and summaries of qualitative themes.
