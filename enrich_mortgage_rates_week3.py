from pathlib import Path
import gc
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week3"

REPORTS.mkdir(parents=True, exist_ok=True)

FRED_URL = (
    "https://fred.stlouisfed.org/graph/"
    "fredgraph.csv?id=MORTGAGE30US"
)

print("Fetching MORTGAGE30US from FRED...")
mortgage = pd.read_csv(FRED_URL)

required = {"observation_date", "MORTGAGE30US"}
if not required.issubset(mortgage.columns):
    raise ValueError(
        f"Unexpected FRED columns: {list(mortgage.columns)}"
    )

mortgage = mortgage.rename(columns={
    "observation_date": "date",
    "MORTGAGE30US": "rate_30yr_fixed"
})

mortgage["date"] = pd.to_datetime(
    mortgage["date"], errors="coerce"
)
mortgage["rate_30yr_fixed"] = pd.to_numeric(
    mortgage["rate_30yr_fixed"], errors="coerce"
)

if mortgage["date"].isna().any():
    raise ValueError("FRED contains unparseable observation dates.")

print(
    "Weekly observations with missing rates:",
    int(mortgage["rate_30yr_fixed"].isna().sum())
)

mortgage["year_month"] = (
    mortgage["date"].dt.to_period("M").astype(str)
)

monthly = (
    mortgage.groupby("year_month", as_index=False)
    .agg(
        rate_30yr_fixed=("rate_30yr_fixed", "mean"),
        weekly_observations=("rate_30yr_fixed", "count")
    )
)

if monthly["year_month"].duplicated().any():
    raise ValueError("Monthly rate table has duplicate month keys.")

monthly.to_csv(
    PROCESSED / "mortgage_rates_monthly.csv", index=False
)

jobs = [
    (
        "Sold",
        "sold_residential_structured.csv",
        "CloseDate",
        "sold_residential_with_rates.csv"
    ),
    (
        "Listings",
        "listings_residential_structured.csv",
        "ListingContractDate",
        "listings_residential_with_rates.csv"
    )
]

validation = []

for label, input_name, date_field, output_name in jobs:
    print(f"\n--- {label.upper()} ---")
    df = pd.read_csv(PROCESSED / input_name, low_memory=False)
    before = len(df)

    if "rate_30yr_fixed" in df.columns:
        raise ValueError(
            f"{label}: input already contains rate_30yr_fixed."
        )

    source_dates = df[date_field].astype("string")
    missing = (
        source_dates.isna()
        | source_dates.str.strip().eq("").fillna(False)
    )

    parsed = pd.to_datetime(
        source_dates, errors="coerce", format="mixed"
    )
    parse_failures = int((~missing & parsed.isna()).sum())

    df["year_month"] = parsed.dt.to_period("M").astype("string")

    enriched = df.merge(
        monthly[["year_month", "rate_30yr_fixed"]],
        on="year_month",
        how="left",
        validate="many_to_one",
        indicator="_rate_merge"
    )

    after = len(enriched)
    if before != after:
        raise ValueError(f"{label}: row count changed during merge.")

    missing_rates = int(enriched["rate_30yr_fixed"].isna().sum())
    unmatched_rows = int(
        enriched["_rate_merge"].eq("left_only").sum()
    )

    if missing_rates:
        unmatched = (
            enriched.loc[
                enriched["rate_30yr_fixed"].isna(), "year_month"
            ]
            .fillna("<MISSING_DATE>")
            .value_counts()
            .rename_axis("year_month")
            .reset_index(name="rows_without_rate")
        )
        unmatched.to_csv(
            REPORTS / f"{label.lower()}_unmatched_months.csv",
            index=False
        )

    enriched = enriched.drop(columns="_rate_merge")
    enriched.to_csv(PROCESSED / output_name, index=False)

    validation.append({
        "dataset": label,
        "join_date_field": date_field,
        "rows_before": before,
        "rows_after": after,
        "missing_blank_dates": int(missing.sum()),
        "date_parse_failures": parse_failures,
        "unmatched_join_rows": unmatched_rows,
        "null_mortgage_rates": missing_rates,
        "validation_passed": missing_rates == 0
    })

    print(f"Rows before merge: {before:,}")
    print(f"Rows after merge:  {after:,}")
    print(f"Missing/blank dates: {int(missing.sum())}")
    print(f"Date parsing failures: {parse_failures}")
    print(f"Unmatched join rows: {unmatched_rows}")
    print(f"Null mortgage rates: {missing_rates}")
    print(f"Saved: {PROCESSED / output_name}")

    del df, enriched
    gc.collect()

validation_df = pd.DataFrame(validation)
validation_df.to_csv(
    REPORTS / "mortgage_merge_validation.csv", index=False
)

if validation_df["validation_passed"].all():
    print("\nWEEK 3 MORTGAGE ENRICHMENT COMPLETE — VALIDATION PASSED")
else:
    print("\nENRICHMENT SAVED — VALIDATION NEEDS REVIEW")
    print("Do not fill missing rates arbitrarily; inspect the reports.")