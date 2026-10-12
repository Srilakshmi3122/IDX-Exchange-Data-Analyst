# Python workflow

Run these scripts in order:

1. combine_week1_data.py — combines monthly MLS files and filters
   Listings and Sold records to Residential properties.
2. structure_validate_week2.py — profiles dataset structure,
   missing values, numeric distributions, and duplicate-name columns.
3. enrich_mortgage_rates_week3.py — calculates monthly FRED mortgage
   rates and joins them to both datasets with validation checks.

## Local folder structure

Place the code folder inside the project folder, alongside:

- data_raw/listed/
- data_raw/sold/
- data_processed/

Reports are generated locally in the reports folder.

The Week 1 script currently uses a local OneDrive/Desktop project path.
The Week 2 and Week 3 scripts locate the project relative to their files.

## Current scope

The workflow has been run on January 2024–April 2026 MLS extracts.
Additional source months still need to be obtained and processed.

## Confidentiality

MLS source files and generated datasets are confidential.
They are not included in this public repository.
Do not commit credentials, internal documents, or confidential outputs.

MLS source files and generated datasets are confidential.
They are not included in this public repository.
Do not commit credentials, internal documents, or confidential outputs.
