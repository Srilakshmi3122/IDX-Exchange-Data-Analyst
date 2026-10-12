from pathlib import Path
import gc
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data_raw"
PROCESSED = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week2"
PLOTS = REPORTS / "plots"

REPORTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

NUMERIC_FIELDS = [
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres", "BedroomsTotal",
    "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt"
]

CORE_FIELDS = set(NUMERIC_FIELDS + [
    "ListingKey", "ListingId", "PropertyType", "PropertySubType",
    "ListingContractDate", "PurchaseContractDate", "CloseDate",
    "ContractStatusChangeDate", "CountyOrParish", "City",
    "PostalCode", "StateOrProvince", "MLSAreaMajor",
    "Latitude", "Longitude", "MlsStatus",
    "ListOfficeName", "BuyerOfficeName",
    "ListAgentFirstName", "ListAgentLastName", "ListAgentFullName",
    "BuyerAgentMlsId", "BuyerAgentFirstName", "BuyerAgentLastName"
])

DATASETS = [
    ("listings", "listed", "combined_listings_residential.csv"),
    ("sold", "sold", "combined_sold_residential.csv")
]


def original_property_types(folder):
    totals = pd.Series(dtype="int64")

    for path in sorted(folder.glob("*.csv")):
        for chunk in pd.read_csv(
            path,
            usecols=["PropertyType"],
            dtype="string",
            chunksize=100000
        ):
            counts = (
                chunk["PropertyType"]
                .fillna("<MISSING>")
                .value_counts()
            )
            totals = totals.add(counts, fill_value=0)

    report = totals.rename_axis("PropertyType").reset_index(name="rows")
    report["rows"] = report["rows"].astype("int64")

    total = report["rows"].sum()
    report["percentage"] = (
        report["rows"] / total * 100 if total else 0
    )

    return report.sort_values("rows", ascending=False)


def numeric_values(series):
    return pd.to_numeric(series, errors="coerce")


def inspect_dataset(label, raw_folder, filename):
    print(f"\n--- {label.upper()} ---")

    path = PROCESSED / filename
    df = pd.read_csv(path, low_memory=False)

    starting_rows = len(df)
    print(f"Input rows: {starting_rows:,}")
    print(f"Input columns: {len(df.columns)}")

    input_types = (
        df["PropertyType"]
        .fillna("<MISSING>")
        .astype(str)
        .value_counts()
        .rename_axis("PropertyType")
        .reset_index(name="rows")
    )
    input_types.to_csv(
        REPORTS / f"{label}_input_property_types.csv", index=False
    )

    residential_mask = (
        df["PropertyType"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("residential")
        .fillna(False)
    )
    df = df.loc[residential_mask].copy()

    print(f"Residential rows: {len(df):,}")
    if df.empty:
        raise ValueError(f"{label}: no Residential records found.")

    types = original_property_types(RAW / raw_folder)
    types.to_csv(
        REPORTS / f"{label}_original_property_types.csv", index=False
    )

    profile = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "null_count": [int(df[c].isna().sum()) for c in df.columns],
        "null_percentage": [
            float(df[c].isna().mean() * 100) for c in df.columns
        ]
    })

    profile["above_90_percent_null"] = (
        profile["null_percentage"] > 90
    )
    profile["field_group"] = profile["column"].apply(
        lambda c: (
            "duplicate-name variant: review"
            if c.endswith(".1")
            else "core market/identifier"
            if c in CORE_FIELDS
            else "other attribute/metadata: review"
        )
    )
    profile["proposed_action"] = profile.apply(
        lambda r: (
            "retain core field"
            if r["column"] in CORE_FIELDS
            else "review before any removal"
            if r["above_90_percent_null"]
            or r["column"].endswith(".1")
            else "retain for now; review relevance"
        ),
        axis=1
    )

    profile.sort_values(
        "null_percentage", ascending=False
    ).to_csv(REPORTS / f"{label}_column_profile.csv", index=False)

    pair_reports = []
    for column in df.columns:
        if not column.endswith(".1"):
            continue

        base = column[:-2]
        if base not in df.columns:
            continue

        left = df[base].astype("string").str.strip()
        right = df[column].astype("string").str.strip()
        both_present = left.notna() & right.notna()
        different = (
            both_present & left.ne(right).fillna(False)
        )

        pair_reports.append({
            "base_column": base,
            "variant_column": column,
            "base_nulls": int(left.isna().sum()),
            "variant_nulls": int(right.isna().sum()),
            "both_present": int(both_present.sum()),
            "text_comparison_differences": int(different.sum()),
            "note": "Review differences; formatting can affect comparison."
        })

    if pair_reports:
        pd.DataFrame(pair_reports).to_csv(
            REPORTS / f"{label}_duplicate_column_review.csv",
            index=False
        )

    summaries = []
    for field in NUMERIC_FIELDS:
        if field not in df.columns:
            print(f"Missing expected numeric field: {field}")
            continue

        values = numeric_values(df[field])
        nonempty = (
            df[field].notna()
            & df[field].astype("string").str.strip().ne("").fillna(False)
        )
        conversion_failures = int((nonempty & values.isna()).sum())
        valid = values.dropna()

        row = {
            "field": field,
            "numeric_count": int(valid.size),
            "numeric_conversion_failures": conversion_failures,
            "min": valid.min(),
            "max": valid.max(),
            "mean": valid.mean(),
            "median": valid.median()
        }

        for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
            row[f"p{int(q * 100):02d}"] = valid.quantile(q)

        summaries.append(row)

        if not valid.empty:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].hist(valid, bins=60)
            axes[0].set_title(f"{field}: histogram")
            axes[0].set_xlabel(field)
            axes[0].set_ylabel("Records")
            axes[1].boxplot(valid, vert=False)
            axes[1].set_title(f"{field}: boxplot")
            axes[1].set_xlabel(field)
            fig.suptitle(f"{label.title()} — full numeric distribution")
            fig.tight_layout()
            fig.savefig(
                PLOTS / f"{label}_{field}.png", dpi=140
            )
            plt.close(fig)

    pd.DataFrame(summaries).to_csv(
        REPORTS / f"{label}_numeric_summary.csv", index=False
    )

    output = PROCESSED / f"{label}_residential_structured.csv"
    df.to_csv(output, index=False)

    high_missing = profile.loc[
        profile["above_90_percent_null"], "column"
    ].tolist()

    print(f"Columns above 90% null: {len(high_missing)}")
    print("High-missing columns:", ", ".join(high_missing) or "None")
    print(f"Duplicate-name column pairs reviewed: {len(pair_reports)}")
    print(f"Saved structured dataset: {output}")
    print("No columns dropped, missing values filled, or outliers removed.")

    summary = {
        "dataset": label,
        "input_rows": starting_rows,
        "residential_rows": len(df),
        "columns": len(df.columns),
        "columns_above_90_percent_null": len(high_missing),
        "duplicate_name_pairs": len(pair_reports)
    }

    del df
    gc.collect()
    return summary


results = []
for label, raw_folder, filename in DATASETS:
    results.append(inspect_dataset(label, raw_folder, filename))

pd.DataFrame(results).to_csv(
    REPORTS / "run_summary.csv", index=False
)

print(f"\nReports saved to: {REPORTS}")
print("WEEK 2 PROFILING RUN COMPLETE")
