from pathlib import Path
import gc
import pandas as pd

# Project paths: this script belongs inside the project's code folder.
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week4"
REPORTS.mkdir(parents=True, exist_ok=True)

DATE_FIELDS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

NUMERIC_FIELDS = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt",
    "Latitude",
    "Longitude",
]

DATASETS = [
    ("listings", "listings_residential_with_rates.csv"),
    ("sold", "sold_residential_with_rates.csv"),
]

run_results = []

for label, filename in DATASETS:
    print(f"\n--- {label.upper()} ---", flush=True)

    input_path = DATA / filename
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read identifiers as strings to avoid numeric conversion.
    df = pd.read_csv(
        input_path,
        dtype={
            "PostalCode": "string",
            "ListingKey": "string",
            "ListingId": "string",
            "ListingKeyNumeric": "string",
        },
        low_memory=False,
    )

    rows_before = len(df)
    original_types = df.dtypes.astype(str)
    conversion_results = []
    created_flags = []

    required = DATE_FIELDS + NUMERIC_FIELDS + [
        "StateOrProvince",
        "rate_30yr_fixed",
    ]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    if any(f"{c}_raw" in df.columns for c in DATE_FIELDS + NUMERIC_FIELDS):
        raise ValueError("Input already contains raw backup columns.")

    print(f"Input rows: {rows_before:,}", flush=True)
    print("Converting dates and numeric fields...", flush=True)

    for field in DATE_FIELDS + NUMERIC_FIELDS:
        original = df[field].copy()
        df[f"{field}_raw"] = original

        text = original.astype("string")
        blank = (
            text.isna()
            | text.str.strip().eq("").fillna(False)
        )

        if field in DATE_FIELDS:
            # Offset-aware timestamps are normalized to UTC.
            # Naive timestamps are assumed UTC for this review pass.
            converted = pd.to_datetime(
                original,
                errors="coerce",
                format="mixed",
                utc=True,
            )
        else:
            converted = pd.to_numeric(original, errors="coerce")

        failure = (~blank & converted.isna()).fillna(False)
        flag_name = f"{field}_parse_failure_flag"
        df[flag_name] = failure
        created_flags.append(flag_name)
        df[field] = converted

        conversion_results.append({
            "field": field,
            "source_missing_or_blank": int(blank.sum()),
            "conversion_failures": int(failure.sum()),
            "result_nulls": int(converted.isna().sum()),
            "result_dtype": str(converted.dtype),
        })

    # Date comparisons retain missing results when either date is absent.
    date_rules = [
        (
            "listing_after_close_flag",
            "ListingContractDate",
            "CloseDate",
        ),
        (
            "purchase_after_close_flag",
            "PurchaseContractDate",
            "CloseDate",
        ),
        (
            "listing_after_purchase_flag",
            "ListingContractDate",
            "PurchaseContractDate",
        ),
    ]

    for flag_name, first, second in date_rules:
        comparable = df[first].notna() & df[second].notna()
        result = (df[first] > df[second]).astype("boolean")
        df[flag_name] = result.where(comparable, pd.NA)
        created_flags.append(flag_name)

    # Any observed reversal is a violation.
    # No observed reversal with incomplete dates remains unknown.
    timeline_columns = [rule[0] for rule in date_rules]
    observed_reversal = (
        df[timeline_columns].fillna(False).any(axis=1)
    )
    complete_timeline = df[[
        "ListingContractDate",
        "PurchaseContractDate",
        "CloseDate",
    ]].notna().all(axis=1)

    timeline = pd.Series(pd.NA, index=df.index, dtype="boolean")
    timeline.loc[complete_timeline] = False
    timeline.loc[observed_reversal] = True
    df["negative_timeline_flag"] = timeline
    created_flags.append("negative_timeline_flag")

    numeric_rules = [
        ("invalid_close_price_flag", "ClosePrice", "nonpositive"),
        ("invalid_living_area_flag", "LivingArea", "nonpositive"),
        ("negative_dom_flag", "DaysOnMarket", "negative"),
        ("negative_bedrooms_flag", "BedroomsTotal", "negative"),
        (
            "negative_bathrooms_flag",
            "BathroomsTotalInteger",
            "negative",
        ),
    ]

    for flag_name, field, rule in numeric_rules:
        values = df[field]
        condition = values.le(0) if rule == "nonpositive" else values.lt(0)
        df[flag_name] = (
            condition.astype("boolean").where(values.notna(), pd.NA)
        )
        created_flags.append(flag_name)

    lat = df["Latitude"]
    lon = df["Longitude"]
    both_coordinates = lat.notna() & lon.notna()

    df["missing_coordinates_flag"] = ~both_coordinates

    df["zero_coordinates_flag"] = (
        lat.eq(0) | lon.eq(0)
    ).astype("boolean").where(both_coordinates, pd.NA)

    df["positive_longitude_flag"] = (
        lon.gt(0).astype("boolean").where(lon.notna(), pd.NA)
    )

    df["invalid_global_coordinates_flag"] = (
        ~lat.between(-90, 90) | ~lon.between(-180, 180)
    ).astype("boolean").where(both_coordinates, pd.NA)

    # Broad screening rectangle, not a California boundary test.
    df["outside_ca_screening_bounds_flag"] = (
        ~(lat.between(32, 42.5) & lon.between(-125, -114))
    ).astype("boolean").where(both_coordinates, pd.NA)

    state = (
        df["StateOrProvince"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    state_present = state.notna() & state.ne("").fillna(False)
    df["non_ca_state_flag"] = (
        (~state.isin(["CA", "CALIFORNIA"]))
        .astype("boolean")
        .where(state_present, pd.NA)
    )

    created_flags.extend([
        "missing_coordinates_flag",
        "zero_coordinates_flag",
        "positive_longitude_flag",
        "invalid_global_coordinates_flag",
        "outside_ca_screening_bounds_flag",
        "non_ca_state_flag",
    ])

    flag_report = pd.DataFrame([
        {
            "flag": name,
            "rows_flagged": int(df[name].fillna(False).sum()),
            "rows_not_assessable": int(df[name].isna().sum()),
        }
        for name in created_flags
    ])

    flag_report.to_csv(
        REPORTS / f"{label}_quality_flags.csv",
        index=False,
    )

    pd.DataFrame(conversion_results).to_csv(
        REPORTS / f"{label}_type_conversions.csv",
        index=False,
    )

    pd.DataFrame({
        "field": original_types.index,
        "dtype_before": original_types.values,
        "dtype_after": [
            str(df[c].dtype) for c in original_types.index
        ],
    }).to_csv(
        REPORTS / f"{label}_data_types.csv",
        index=False,
    )

    rows_after = len(df)
    if rows_before != rows_after:
        raise ValueError("Unexpected row-count change.")

    output_path = DATA / f"{label}_residential_cleaning_review.csv"
    print("Saving review dataset...", flush=True)
    df.to_csv(output_path, index=False)

    run_results.append({
        "dataset": label,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "rows_with_any_flag": int(
            df[created_flags].fillna(False).any(axis=1).sum()
        ),
        "null_mortgage_rates": int(
            df["rate_30yr_fixed"].isna().sum()
        ),
    })

    print(f"Rows before: {rows_before:,}")
    print(f"Rows after:  {rows_after:,}")
    print(flag_report.to_string(index=False))
    print(f"Saved: {output_path}", flush=True)

    del df
    gc.collect()

pd.DataFrame(run_results).to_csv(
    REPORTS / "cleaning_run_summary.csv",
    index=False,
)

print("\nINITIAL CLEANING AND FLAGGING COMPLETE")
print("No records or original columns were removed.")
print("Missing values were not imputed.")
print("Review outputs before finalizing cleaning decisions.")