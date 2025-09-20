# AMR Data Extraction

This repository extracts structured quantitative and qualitative data about SSI and AMR from OCR-parsed articles.

## What's new

Optional extensions and extraction checklist were added:

- Study: country_iso3, ownership_type, bed_capacity, surgical_volume_per_year
- Timeframe: data_collection_start_month, data_collection_end_month
- SSI incidence: ssi_type, readmission_window_days, reoperation_definition_text
- AMR proportions: MDR/XDR, ESBL, carbapenemase, MRSA method, gene markers, MIC/zone units and breakpoints
- Qualitative context: method, participants_n, analysis_approach, casp_appraisal

These are non-mandatory—captured only if present.

## Must-have vs Nice-to-have

Must-have (quantitative):

- Study identifiers; design; timeframe (start/end years)
- SSI incidence: events, denominator, denominator_type, follow-up window
- AMR: pathogen, antibiotic, n_tested, n_resistant, AST method, breakpoint standard/version, specimen_type
- Subgroups: facility_level, care_setting, surgical_specialty

Nice-to-have:

- SSI subtype counts; ascertainment and follow-up method
- Risk context: ASA, wound class, HIV/diabetes %
- AMR advanced: MDR/XDR, ESBL, carbapenemase, genotypes, MRSA method
- Breakpoint numeric context (MIC/zone units and thresholds)
- Lab QA; clinical outcomes (mortality, reoperation, readmission, LOS)
- Economic: perspective, currency_code, price_year, amount, exchange rate / PPP
- Dedup keys: ownership, bed capacity, surgical volume, exact period
- Qualitative method context (method, N, analysis, CASP)

## How to run

1. Generate extraction prompts or call LLM directly:

- Dry-run (compose prompts):
  scripts/run_extraction.py --input-dir data/ocr --prompt prompts/ocr_to_dataset_prompt.txt --out-dir outputs/extracted --quant-schema data_schemas/quantitative_schema.json --qual-schema data_schemas/qualitative_schema.json --dry-run

- Live (requires OPENAI_API_KEY):
  scripts/run_extraction.py --input-dir data/ocr --prompt prompts/ocr_to_dataset_prompt.txt --out-dir outputs/extracted --quant-schema data_schemas/quantitative_schema.json --qual-schema data_schemas/qualitative_schema.json

Performance and control flags (optional):

- --workers 4 process files in parallel (tune to avoid rate limits)
- --compact-ocr drop heavy OCR keys like bbox/images/raw/tokens to reduce tokens
- --filter-ocr-text-only keep only text-bearing fields (text, headings, tables)
- --max-prompt-chars 120000 hard cap on OCR JSON string length in the prompt

Using Azure OpenAI (optional):

- You can either set environment variables or pass flags. A .env file at repo root is auto-loaded if present; lines in the form NAME=value or export NAME=value are supported.
- Environment variables (aliases supported):
  - AZURE_OPENAI_ENDPOINT (alias: OPENAI_AZURE_ENDPOINT)
  - AZURE_OPENAI_API_KEY (alias: AZURE_OPENAI_KEY)
  - AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT_NAME
  - AZURE_OPENAI_API_VERSION (defaults to 2024-02-15-preview)
- Flags:
  - --use-azure
  - --azure-endpoint https://<resource>.openai.azure.com/
  - --azure-api-key <key>
  - --azure-deployment <deployment-name>
  - --azure-api-version 2024-02-15-preview
    Notes: When --use-azure is set or Azure env vars are present, the script switches to the Azure client. On Azure, the deployment name selects the model.

2. Validate extracted JSONs:

   scripts/validate_and_prepare.py --in-dir outputs/extracted --quant-schema data_schemas/quantitative_schema.json --qual-schema data_schemas/qualitative_schema.json

3. Aggregate to CSVs:

scripts/aggregate_to_tabular.py \
 --in-dir outputs/extracted \
 --out-processed outputs/processed/processed.csv \
 --out-ssi outputs/processed/ssi_incidence_long.csv \
 --out-amr outputs/processed/amr_proportions_long.csv \
 --out-codes outputs/processed/qual_codes.csv \
 --out-mortality outputs/processed/mortality_long.csv \
 --out-readmissions outputs/processed/readmissions_long.csv \
 --out-los outputs/processed/length_of_stay_long.csv \
 --out-costs outputs/processed/costs_long.csv \
 --out-quant-thematic outputs/processed/quant_thematic.csv \
 --out-thematic-all outputs/processed/thematic_all.csv \
 --out-ssi-stratified outputs/processed/ssi_stratified.csv \
 --out-amr-stratified outputs/processed/amr_stratified.csv

All commands are Python scripts; run with your environment's Python interpreter.
