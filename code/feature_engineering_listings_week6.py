from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week6"

INPUT = DATA / "listings_residential_analysis.csv"
OUTPUT = DATA / "listings_residential_features.csv"

FEATURE_COLUMNS = [
    "listing_year",
    "listing_month",
    "listing_yrmo",
    "listing_to_contract_days",
]


def parse_date(series):
    text = series.str.strip()
    return pd.to_datetime(
        text.mask(text.eq("")),
        errors="raise",
        format="mixed",
        utc=True,
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    print("1/6 Reading Listings analysis dataset...", flush=True)

    # Preserve existing fields, identifiers, flags, and blank values.
    df = pd.read_csv(
        INPUT,
        dtype="string",
        keep_default_na=False,
    )

    rows_before = len(df)
    columns_before = len(df.columns)

    print(f"Loaded {rows_before:,} records.", flush=True)

    required = [
        "ListingKey",
        "ListingContractDate",
        "PurchaseContractDate",
        "CountyOrParish",
        "year_month",
        "rate_30yr_fixed",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required fields missing: {missing}")

    existing = [c for c in FEATURE_COLUMNS if c in df.columns]
    if existing:
        raise ValueError(f"Feature columns already exist: {existing}")

    print("2/6 Parsing dates for calculations...", flush=True)

    print("  Parsing ListingContractDate...", flush=True)
    listing_date = parse_date(df["ListingContractDate"])

    print("  Parsing PurchaseContractDate...", flush=True)
    purchase_date = parse_date(df["PurchaseContractDate"])

    print("3/6 Creating Listings features...", flush=True)

    df["listing_year"] = listing_date.dt.year.astype("Int64")
    df["listing_month"] = listing_date.dt.month.astype("Int64")
    df["listing_yrmo"] = (
        listing_date.dt.strftime("%Y-%m").astype("string")
    )

    # Elapsed days may include fractions when timestamps contain times.
    elapsed_days = (
        purchase_date - listing_date
    ).dt.total_seconds() / 86400

    df["listing_to_contract_days"] = elapsed_days.where(
        elapsed_days.ge(0)
    )

    print("4/6 Checking features and preparing reports...", flush=True)

    if len(df) != rows_before:
        raise ValueError("Unexpected row-count change.")

    durations = df["listing_to_contract_days"].dropna()
    if durations.lt(0).any():
        raise ValueError("Negative calculated durations remain.")

    coverage_rows = []

    for field in FEATURE_COLUMNS:
        populated = int(df[field].notna().sum())

        coverage_rows.append({
            "feature": field,
            "total_records": rows_before,
            "populated_records": populated,
            "missing_records": rows_before - populated,
        })

    coverage = pd.DataFrame(coverage_rows)

    county_data = pd.DataFrame({
        "CountyOrParish": (
            df["CountyOrParish"].str.strip().replace("", pd.NA)
        ),
        "listing_to_contract_days": df["listing_to_contract_days"],
    })

    county_summary = (
        county_data.groupby("CountyOrParish", dropna=False)
        .agg(
            record_count=("listing_to_contract_days", "size"),
            valid_duration_count=("listing_to_contract_days", "count"),
            median_listing_to_contract_days=(
                "listing_to_contract_days",
                "median",
            ),
        )
        .reset_index()
    )

    if int(county_summary["record_count"].sum()) != rows_before:
        raise ValueError("County summary does not account for every row.")

    sample_columns = [
        "ListingKey",
        "ListingContractDate",
        "PurchaseContractDate",
    ] + FEATURE_COLUMNS

    print("5/6 Saving Listings feature dataset...", flush=True)
    df.to_csv(OUTPUT, index=False)

    print("6/6 Saving local reports...", flush=True)

    coverage.to_csv(
        REPORTS / "listings_feature_coverage.csv",
        index=False,
    )

    county_summary.to_csv(
        REPORTS / "listings_county_summary.csv",
        index=False,
    )

    df[sample_columns].head(10).to_csv(
        REPORTS / "listings_feature_sample.csv",
        index=False,
    )

    pd.DataFrame([{
        "dataset": "listings",
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_before": columns_before,
        "columns_after": len(df.columns),
        "features_added": len(FEATURE_COLUMNS),
        "output_file": OUTPUT.name,
    }]).to_csv(
        REPORTS / "listings_feature_run_summary.csv",
        index=False,
    )

    print("\nLISTINGS FEATURE ENGINEERING COMPLETE")
    print(f"Records retained: {len(df):,}")
    print(f"Features added: {len(FEATURE_COLUMNS)}")

    print("\nFEATURE COVERAGE")
    print(coverage.to_string(index=False))

    print("\nInput file was not overwritten.")
    print("Original fields and mortgage-merge year_month were retained.")
    print("County counts represent records, not verified unique listings.")
    print("Duration summaries use only assessable nonnegative durations.")
    print("All CSV outputs must remain local.")


if __name__ == "__main__":
    main()