from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week6"

INPUT = DATA / "sold_residential_analysis.csv"
OUTPUT = DATA / "sold_residential_features.csv"

DATE_FIELDS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

NUMERIC_FIELDS = [
    "ClosePrice",
    "OriginalListPrice",
    "LivingArea",
    "DaysOnMarket",
]

FEATURE_COLUMNS = [
    "price_ratio",
    "close_to_original_list_ratio",
    "price_per_sqft",
    "close_year",
    "close_month",
    "close_yrmo",
    "listing_year",
    "listing_month",
    "listing_yrmo",
    "listing_to_contract_days",
    "contract_to_close_days",
]


def positive_ratio(numerator, denominator):
    valid = (
        numerator.gt(0)
        & denominator.gt(0)
        & np.isfinite(numerator)
        & np.isfinite(denominator)
    ).fillna(False)

    result = numerator.where(valid) / denominator.where(valid)
    return result.where(np.isfinite(result))


def nonnegative_duration(start, end):
    # Elapsed days, including fractions if timestamps contain times.
    days = (end - start).dt.total_seconds() / 86400
    return days.where(days.ge(0))


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    print("1/6 Reading Sold analysis dataset...", flush=True)

    # Preserve existing CSV fields as text, including identifiers.
    df = pd.read_csv(
        INPUT,
        dtype="string",
        keep_default_na=False,
    )

    rows_before = len(df)
    print(f"Loaded {rows_before:,} records.", flush=True)

    required = DATE_FIELDS + NUMERIC_FIELDS + [
        "ListingKey",
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

    print("2/6 Parsing calculation inputs...", flush=True)

    numbers = {}
    dates = {}

    for field in NUMERIC_FIELDS:
        text = df[field].str.strip()
        numbers[field] = pd.to_numeric(
            text.mask(text.eq("")),
            errors="raise",
        )

    for field in DATE_FIELDS:
        print(f"  Parsing {field}...", flush=True)
        text = df[field].str.strip()
        dates[field] = pd.to_datetime(
            text.mask(text.eq("")),
            errors="raise",
            format="mixed",
            utc=True,
        )

    print("3/6 Calculating market features...", flush=True)

    df["price_ratio"] = positive_ratio(
        numbers["ClosePrice"],
        numbers["OriginalListPrice"],
    )

    # The handbook gives these two names the same formula.
    df["close_to_original_list_ratio"] = df["price_ratio"]

    # Interpretation assumes LivingArea is measured in square feet.
    df["price_per_sqft"] = positive_ratio(
        numbers["ClosePrice"],
        numbers["LivingArea"],
    )

    for prefix, field in [
        ("close", "CloseDate"),
        ("listing", "ListingContractDate"),
    ]:
        df[f"{prefix}_year"] = dates[field].dt.year.astype("Int64")
        df[f"{prefix}_month"] = dates[field].dt.month.astype("Int64")
        df[f"{prefix}_yrmo"] = (
            dates[field].dt.strftime("%Y-%m").astype("string")
        )

    df["listing_to_contract_days"] = nonnegative_duration(
        dates["ListingContractDate"],
        dates["PurchaseContractDate"],
    )

    df["contract_to_close_days"] = nonnegative_duration(
        dates["PurchaseContractDate"],
        dates["CloseDate"],
    )

    print("4/6 Checking features and preparing reports...", flush=True)

    if len(df) != rows_before:
        raise ValueError("Unexpected row-count change.")

    for field in [
        "price_ratio",
        "close_to_original_list_ratio",
        "price_per_sqft",
    ]:
        values = df[field].dropna()
        if values.le(0).any() or not np.isfinite(values).all():
            raise ValueError(f"Invalid calculated ratio: {field}")

    for field in [
        "listing_to_contract_days",
        "contract_to_close_days",
    ]:
        if df[field].dropna().lt(0).any():
            raise ValueError(f"Negative calculated duration: {field}")

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
        "ClosePrice": numbers["ClosePrice"].where(
            numbers["ClosePrice"].gt(0)
            & np.isfinite(numbers["ClosePrice"])
        ),
        "DaysOnMarket": numbers["DaysOnMarket"].where(
            numbers["DaysOnMarket"].ge(0)
            & np.isfinite(numbers["DaysOnMarket"])
        ),
        "price_per_sqft": df["price_per_sqft"],
        "price_ratio": df["price_ratio"],
    })

    county_summary = (
        county_data.groupby("CountyOrParish", dropna=False)
        .agg(
            record_count=("ClosePrice", "size"),
            valid_close_price_count=("ClosePrice", "count"),
            median_close_price=("ClosePrice", "median"),
            valid_dom_count=("DaysOnMarket", "count"),
            average_days_on_market=("DaysOnMarket", "mean"),
            valid_ppsf_count=("price_per_sqft", "count"),
            median_price_per_sqft=("price_per_sqft", "median"),
            valid_ratio_count=("price_ratio", "count"),
            average_price_ratio=("price_ratio", "mean"),
        )
        .reset_index()
    )

    sample_columns = [
        "ListingKey",
        "ClosePrice",
        "OriginalListPrice",
        "LivingArea",
        "DaysOnMarket",
        "ListingContractDate",
        "PurchaseContractDate",
        "CloseDate",
    ] + FEATURE_COLUMNS

    print("5/6 Saving feature dataset...", flush=True)
    df.to_csv(OUTPUT, index=False)

    print("6/6 Saving local reports...", flush=True)

    coverage.to_csv(
        REPORTS / "sold_feature_coverage.csv",
        index=False,
    )

    county_summary.to_csv(
        REPORTS / "sold_county_summary.csv",
        index=False,
    )

    df[sample_columns].head(10).to_csv(
        REPORTS / "sold_feature_sample.csv",
        index=False,
    )

    pd.DataFrame([{
        "dataset": "sold",
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_before": len(df.columns) - len(FEATURE_COLUMNS),
        "columns_after": len(df.columns),
        "features_added": len(FEATURE_COLUMNS),
        "output_file": OUTPUT.name,
    }]).to_csv(
        REPORTS / "sold_feature_run_summary.csv",
        index=False,
    )

    print("\nSOLD FEATURE ENGINEERING COMPLETE")
    print(f"Records retained: {len(df):,}")
    print(f"Features added: {len(FEATURE_COLUMNS)}")
    print("\nFEATURE COVERAGE")
    print(coverage.to_string(index=False))
    print("\nInput file was not overwritten.")
    print("Existing flags and mortgage-merge fields were retained.")
    print("All CSV outputs must remain local.")


if __name__ == "__main__":
    main()