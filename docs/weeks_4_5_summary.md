# Weeks 4–5: Data Cleaning and Preparation

## Objective

Prepare the Residential Listings and Sold datasets for subsequent
market analysis while preserving source review files and documenting
all cleaning decisions.

## Scripts and run order

Run these scripts from the project after the rate-enriched input
datasets are available in data_processed:

1. python code/clean_validate_week4.py
2. python code/prepare_week5.py
3. python code/export_analysis_week5.py
4. python code/validate_week5_exports.py

The scripts use project-relative paths and require Python with pandas.
The date conversion code uses pandas format="mixed".

## Input files

- data_processed/listings_residential_with_rates.csv
- data_processed/sold_residential_with_rates.csv

MLS datasets are confidential and are not included with this code upload.
Dataset submission must use a program-approved location.

## Transformations

- Converted four date fields to UTC-aware datetime values.
- Converted eleven selected numeric fields.
- Preserved pre-conversion values in raw backup columns in review files.
- Created parsing, date-consistency, numeric, and geographic quality flags.
- Replaced nonpositive ClosePrice and LivingArea values with missing.
- Replaced negative DaysOnMarket, bedroom, and bathroom values with missing.
- Preserved existing missing values without imputation.
- Created reduced analysis exports using documented column selection.
- Retained all 30 quality flags in both analysis exports.
- Preserved full prepared and review files.

## Row and column counts

| Dataset | Input rows | Final analysis rows | Final analysis columns |
|---|---:|---:|---:|
| Listings | 547,162 | 547,162 | 78 |
| Sold | 414,054 | 414,054 | 84 |

No rows were removed.

Both full prepared datasets contained 131 columns.
The reduced exports excluded 53 Listings columns and 47 Sold columns.

Six optional fields were absent from Listings and available in Sold:
latfilled, lonfilled, ListAgentAOR, BuyerAgentAOR,
OriginatingSystemName, and OriginatingSystemSubName.

## Numeric cleaning results

| Field | Listings values changed to missing | Sold values changed to missing |
|---|---:|---:|
| ClosePrice | 0 | 1 |
| LivingArea | 360 | 150 |
| DaysOnMarket | 27 | 48 |
| BedroomsTotal | 0 | 0 |
| BathroomsTotalInteger | 0 | 0 |

These are field-level changes, not deleted records.

## Date consistency review

- Listings: 530 records had at least one observed timeline reversal.
- Sold: 507 records had at least one observed timeline reversal.
- Five flagged Sold examples were inspected; their flags matched
  the displayed date reversals.
- Dates were retained without guessing corrections.
- Incomplete date comparisons remain unassessable where appropriate.

## Geographic quality summary

| Check | Listings flagged | Sold flagged |
|---|---:|---:|
| Missing latitude or longitude | 80,247 | 4,104 |
| Zero latitude or longitude | 64 | 27 |
| Positive longitude | 73 | 31 |
| Globally invalid coordinates | 4 | 2 |
| Outside California screening rectangle | 283 | 89 |
| Non-California state | 155 | 29 |

Categories may overlap and must not be added to estimate unique records.

The screening rectangle is latitude 32 to 42.5 and longitude
-125 to -114. It is not an exact California boundary test.
Coordinates and state values were flagged rather than automatically corrected.

## Validation results

Both saved analysis CSVs passed read-back checks for:

- Expected row and column counts.
- Numeric cleaning rules and expected missing-value counts.
- Parsing of nonblank date values.
- Quality-flag counts and unassessable counts.
- No missing mortgage-rate values.

The preparation script also checked that mortgage-rate values and
quality flags were unchanged before export.

## Output datasets

- data_processed/listings_residential_analysis.csv
- data_processed/sold_residential_analysis.csv

Full audit versions remain available locally:

- listings_residential_cleaning_review.csv
- sold_residential_cleaning_review.csv
- listings_residential_prepared.csv
- sold_residential_prepared.csv

## Supporting reports

Local reports are saved under reports/week4 and reports/week5.

They document conversion results, data types, flag counts,
numeric cleaning, column selection, export counts, and read-back validation.

## Limitations and downstream use

- Dates without timezone information were assumed to be UTC.
- Raw backups reflect values after the initial CSV read, not necessarily
  the exact original CSV text formatting.
- CSV files do not retain pandas data types; downstream scripts must
  explicitly parse dates, numeric fields, and boolean flags.
- Passing validation does not resolve source-date reversals or
  geographic quality issues.
- Mortgage-rate completeness was checked; this phase did not independently
  verify rate ranges or the original merge logic.
- Price metrics require positive, nonmissing inputs.
- Duration metrics must not use missing or negatively ordered date pairs.
- Statistical outlier handling and feature engineering are outside
  this Weeks 4–5 submission.
