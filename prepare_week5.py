from pathlib import Path
import gc
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week5"
REPORTS.mkdir(parents=True, exist_ok=True)

DATASETS = ["listings", "sold"]

RULES = {
    "ClosePrice": "nonpositive",
    "LivingArea": "nonpositive",
    "DaysOnMarket": "negative",
    "BedroomsTotal": "negative",
    "BathroomsTotalInteger": "negative",
}

IDENTIFIERS = [
    "PostalCode",
    "ListingKey",
    "ListingId",
    "ListingKeyNumeric",
]

transformation_results = []
run_results = []

for label in DATASETS:
    print(f"\n--- {label.upper()} ---", flush=True)

    input_path = DATA / f"{label}_residential_cleaning_review.csv"
    output_path = DATA / f"{label}_residential_prepared.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    columns = pd.read_csv(input_path, nrows=0).columns.tolist()

    # Read identifiers and existing raw backups as strings.
    protected = [
        c for c in columns
        if c in IDENTIFIERS or c.endswith("_raw")
    ]

    df = pd.read_csv(
        input_path,
        dtype={c: "string" for c in protected},
        low_memory=False,
    )

    required = list(RULES) + ["rate_30yr_fixed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    rows_before = len(df)
    original_columns = df.columns.tolist()

    flag_columns = [c for c in df.columns if c.endswith("_flag")]
    if not flag_columns:
        raise ValueError("Expected quality flags were not found.")

    flags_before = df[flag_columns].copy(deep=True)
    rates_before = df["rate_30yr_fixed"].copy(deep=True)

    print(f"Input rows: {rows_before:,}", flush=True)

    for field, rule in RULES.items():
        values = pd.to_numeric(df[field], errors="raise")
        missing_before = int(values.isna().sum())

        invalid = (
            values.le(0)
            if rule == "nonpositive"
            else values.lt(0)
        ).fillna(False)

        changed = int(invalid.sum())

        # Keep the record; only the invalid field becomes missing.
        df[field] = values.mask(invalid)

        remaining_invalid = (
            df[field].le(0)
            if rule == "nonpositive"
            else df[field].lt(0)
        ).fillna(False)

        if remaining_invalid.any():
            raise ValueError(f"Invalid values remain in {field}")

        missing_after = int(df[field].isna().sum())
        if missing_after != missing_before + changed:
            raise ValueError(f"Unexpected missing-count change: {field}")

        transformation_results.append({
            "dataset": label,
            "field": field,
            "rule": "<= 0 to missing" if rule == "nonpositive"
                    else "< 0 to missing",
            "missing_before": missing_before,
            "values_changed_to_missing": changed,
            "missing_after": missing_after,
            "dtype_after": str(df[field].dtype),
        })

        print(f"{field}: {changed:,} values set to missing")

    # Validate preservation before writing the prepared output.
    if len(df) != rows_before:
        raise ValueError("Unexpected row-count change.")

    if df.columns.tolist() != original_columns:
        raise ValueError("Unexpected column change.")

    if not df[flag_columns].equals(flags_before):
        raise ValueError("Quality flags changed unexpectedly.")

    if not df["rate_30yr_fixed"].equals(rates_before):
        raise ValueError("Mortgage-rate values changed unexpectedly.")

    null_rates = int(df["rate_30yr_fixed"].isna().sum())
    if null_rates:
        raise ValueError(f"Missing mortgage rates: {null_rates}")

    print("Saving prepared dataset...", flush=True)
    df.to_csv(output_path, index=False)

    run_results.append({
        "dataset": label,
        "rows_before": rows_before,
        "rows_after": len(df),
        "rows_removed": rows_before - len(df),
        "columns_retained": len(df.columns),
        "quality_flags_unchanged": True,
        "mortgage_rates_unchanged": True,
        "null_mortgage_rates": null_rates,
        "output_file": output_path.name,
    })

    print(f"Saved: {output_path.name}", flush=True)

    del df, flags_before, rates_before
    gc.collect()

pd.DataFrame(transformation_results).to_csv(
    REPORTS / "numeric_cleaning_changes.csv",
    index=False,
)

summary = pd.DataFrame(run_results)
summary.to_csv(
    REPORTS / "preparation_run_summary.csv",
    index=False,
)

print("\nPREPARATION COMPLETE")
print(summary.to_string(index=False))
print("\nReview inputs were not overwritten.")
print("Dates, coordinates, flags, and raw backups were retained.")