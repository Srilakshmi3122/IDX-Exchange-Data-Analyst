from pathlib import Path
import gc
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week5"

EXPECTED = {
    "listings": {"rows": 547162, "columns": 78},
    "sold": {"rows": 414054, "columns": 84},
}

RULES = {
    "ClosePrice": "nonpositive",
    "LivingArea": "nonpositive",
    "DaysOnMarket": "negative",
    "BedroomsTotal": "negative",
    "BathroomsTotalInteger": "negative",
}

DATE_FIELDS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

changes = pd.read_csv(REPORTS / "numeric_cleaning_changes.csv")
flag_reports = PROJECT / "reports" / "week4"
results = []

for label, expected in EXPECTED.items():
    print(f"\n--- {label.upper()} ---", flush=True)

    path = DATA / f"{label}_residential_analysis.csv"

    # Keep stored text intact; explicitly parse fields for validation.
    df = pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
    )

    if len(df) != expected["rows"]:
        raise ValueError(f"{label}: unexpected row count.")

    if len(df.columns) != expected["columns"]:
        raise ValueError(f"{label}: unexpected column count.")

    numeric_report = changes.loc[changes["dataset"].eq(label)]

    for field, rule in RULES.items():
        text = df[field].str.strip()
        values = pd.to_numeric(
            text.mask(text.eq("")),
            errors="raise",
        )

        invalid = (
            values.le(0)
            if rule == "nonpositive"
            else values.lt(0)
        ).fillna(False)

        if invalid.any():
            raise ValueError(f"{label}: invalid values remain in {field}.")

        expected_missing = int(
            numeric_report.loc[
                numeric_report["field"].eq(field),
                "missing_after",
            ].iloc[0]
        )

        if int(values.isna().sum()) != expected_missing:
            raise ValueError(f"{label}: wrong missing count for {field}.")

    for field in DATE_FIELDS:
        text = df[field].str.strip()
        pd.to_datetime(
            text.mask(text.eq("")),
            errors="raise",
            format="mixed",
            utc=True,
        )

    rates = pd.to_numeric(
        df["rate_30yr_fixed"].str.strip().replace("", pd.NA),
        errors="raise",
    )

    if rates.isna().any():
        raise ValueError(f"{label}: missing mortgage-rate values.")

    expected_flags = pd.read_csv(
        flag_reports / f"{label}_quality_flags.csv"
    )

    flag_columns = [c for c in df.columns if c.endswith("_flag")]

    if set(flag_columns) != set(expected_flags["flag"]):
        raise ValueError(f"{label}: quality-flag columns do not match.")

    for row in expected_flags.itertuples(index=False):
        values = df[row.flag].str.strip().str.lower()

        if not values.isin(["true", "false", ""]).all():
            raise ValueError(f"{label}: unexpected flag values: {row.flag}")

        if int(values.eq("true").sum()) != row.rows_flagged:
            raise ValueError(f"{label}: flagged count changed: {row.flag}")

        if int(values.eq("").sum()) != row.rows_not_assessable:
            raise ValueError(
                f"{label}: unassessable count changed: {row.flag}"
            )

    results.append({
        "dataset": label,
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_checks": "PASS",
        "date_parsing": "PASS",
        "flag_counts": "PASS",
        "missing_rate_check": "PASS",
    })

    print("All read-back checks passed.", flush=True)

    del df
    gc.collect()

summary = pd.DataFrame(results)
summary.to_csv(
    REPORTS / "analysis_readback_validation.csv",
    index=False,
)

print("\nREAD-BACK VALIDATION COMPLETE")
print(summary.to_string(index=False))