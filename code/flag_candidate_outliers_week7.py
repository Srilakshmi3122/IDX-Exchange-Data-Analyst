from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week7"

CONFIGS = [
    {
        "dataset": "sold",
        "input": "sold_residential_features_school_districts.csv",
        "output": (
            "sold_residential_features_school_districts"
            "_candidate_outliers_week7.csv"
        ),
        "thresholds": "sold_outlier_candidate_iqr.csv",
        "quality": "sold_outlier_input_quality.csv",
        "expected_records": 414054,
        "fields": ["ClosePrice", "LivingArea", "DaysOnMarket"],
    },
    {
        "dataset": "listings",
        "input": "listings_residential_features_school_districts.csv",
        "output": (
            "listings_residential_features_school_districts"
            "_candidate_outliers_week7.csv"
        ),
        "thresholds": "listings_outlier_candidate_iqr.csv",
        "quality": "listings_outlier_input_quality.csv",
        "expected_records": 547162,
        "fields": ["ListPrice", "LivingArea", "DaysOnMarket"],
    },
]


def analyze_field(series, field):
    text = series.str.strip()
    missing = text.eq("")

    values = pd.to_numeric(
        text.mask(missing),
        errors="coerce",
    ).astype("float64")

    parse_failure = ~missing & values.isna()
    finite = values.notna() & np.isfinite(values)
    nonfinite = values.notna() & ~np.isfinite(values)

    if field == "DaysOnMarket":
        invalid = finite & values.lt(0)
    else:
        invalid = finite & values.le(0)

    eligible = finite & ~invalid

    status = pd.Series(
        "eligible",
        index=series.index,
        dtype="string",
    )
    status.loc[missing] = "missing_blank"
    status.loc[parse_failure] = "numeric_parse_failure"
    status.loc[nonfinite] = "nonfinite"
    status.loc[invalid] = "business_rule_invalid"

    counts = {
        "missing_blank": int(missing.sum()),
        "numeric_parse_failures": int(parse_failure.sum()),
        "nonfinite_values": int(nonfinite.sum()),
        "business_rule_invalid": int(invalid.sum()),
        "eligible_for_iqr": int(eligible.sum()),
    }

    return values, eligible, status, counts


def process_dataset(config):
    name = config["dataset"]
    input_path = DATA / config["input"]
    output_path = DATA / config["output"]
    threshold_path = REPORTS / config["thresholds"]
    quality_path = REPORTS / config["quality"]

    for path in [input_path, threshold_path, quality_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output must not overwrite the input.")

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path.name}. "
            "Review it before rerunning; it will not be overwritten."
        )

    print(f"\nReading {name}...", flush=True)

    df = pd.read_csv(
        input_path,
        dtype="string",
        keep_default_na=False,
    )

    original_columns = df.columns.tolist()
    total = len(df)

    if total != config["expected_records"]:
        raise ValueError(
            f"{name}: expected {config['expected_records']:,} "
            f"records, found {total:,}."
        )

    if any(c.startswith("w7_") for c in original_columns):
        raise ValueError(
            f"{name}: existing w7_ columns found. "
            "Use the original unflagged input."
        )

    if not set(config["fields"]).issubset(df.columns):
        raise ValueError(f"{name}: required numeric fields are missing.")

    thresholds = pd.read_csv(threshold_path).set_index("field")
    quality = pd.read_csv(quality_path).set_index("field")

    if not thresholds.index.is_unique or not quality.index.is_unique:
        raise ValueError(f"{name}: duplicate fields in saved reports.")

    any_flag = pd.Series(False, index=df.index)
    all_eligible = pd.Series(True, index=df.index)
    flag_count = pd.Series(0, index=df.index, dtype="int64")
    field_rows = []

    for field in config["fields"]:
        if field not in thresholds.index or field not in quality.index:
            raise ValueError(f"{name}: no saved profile for {field}")

        values, eligible, status, counts = analyze_field(
            df[field],
            field,
        )

        saved = thresholds.loc[field]
        saved_quality = quality.loc[field]

        if int(saved_quality["total_records"]) != total:
            raise ValueError(f"{name}/{field}: profile row count changed.")

        for key, count in counts.items():
            if int(saved_quality[key]) != count:
                raise ValueError(
                    f"{name}/{field}: {key} changed. "
                    "Rerun profiling before flagging."
                )

        population = values.loc[eligible]
        q1 = population.quantile(0.25)
        q3 = population.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        checks = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "candidate_lower_bound": lower,
            "candidate_upper_bound": upper,
        }

        for key, calculated in checks.items():
            if not np.isclose(
                float(saved[key]),
                calculated,
                rtol=1e-12,
                atol=1e-9,
            ):
                raise ValueError(
                    f"{name}/{field}: saved {key} does not match "
                    "the current data. Rerun profiling."
                )

        # Apply the bounds from the saved candidate report.
        lower = float(saved["candidate_lower_bound"])
        upper = float(saved["candidate_upper_bound"])

        below = eligible & values.lt(lower)
        above = eligible & values.gt(upper)
        outside = below | above

        expected_counts = {
            "eligible_records": int(eligible.sum()),
            "eligible_below_lower_bound": int(below.sum()),
            "eligible_above_upper_bound": int(above.sum()),
            "eligible_outside_bounds": int(outside.sum()),
        }

        for key, count in expected_counts.items():
            if int(saved[key]) != count:
                raise ValueError(
                    f"{name}/{field}: {key} differs from "
                    "the saved candidate report."
                )

        candidate = pd.Series(
            pd.NA,
            index=df.index,
            dtype="boolean",
        )
        candidate.loc[eligible] = outside.loc[eligible]

        direction = pd.Series(
            "not_eligible",
            index=df.index,
            dtype="string",
        )
        direction.loc[eligible] = "within_bounds"
        direction.loc[below] = "below_lower_bound"
        direction.loc[above] = "above_upper_bound"

        df[f"w7_{field}_eligibility_status"] = status
        df[f"w7_{field}_iqr_eligible"] = eligible
        df[f"w7_{field}_candidate_iqr_outlier"] = candidate
        df[f"w7_{field}_candidate_iqr_direction"] = direction

        any_flag = any_flag | outside
        all_eligible = all_eligible & eligible
        flag_count = flag_count + outside.astype("int64")

        field_rows.append({
            "dataset": name,
            "field": field,
            "total_records": total,
            "eligible_records": int(eligible.sum()),
            "ineligible_records": int((~eligible).sum()),
            "candidate_lower_bound": lower,
            "candidate_upper_bound": upper,
            "below_lower_bound": int(below.sum()),
            "above_upper_bound": int(above.sum()),
            "candidate_flagged_records": int(outside.sum()),
            "candidate_flagged_pct_of_eligible": (
                100 * outside.sum() / eligible.sum()
            ),
        })

        print(f"  Validated and flagged {field}", flush=True)

    df["w7_all_three_fields_iqr_eligible"] = all_eligible
    df["w7_any_field_candidate_iqr_outlier"] = any_flag
    df["w7_candidate_iqr_flagged_field_count"] = flag_count

    if df.columns[:len(original_columns)].tolist() != original_columns:
        raise ValueError(f"{name}: original column order changed.")

    summary = {
        "dataset": name,
        "input_file": input_path.name,
        "output_file": output_path.name,
        "records_read": total,
        "records_written": len(df),
        "original_columns": len(original_columns),
        "added_columns": len(df.columns) - len(original_columns),
        "all_three_fields_eligible": int(all_eligible.sum()),
        "at_least_one_field_ineligible": int((~all_eligible).sum()),
        "unique_records_with_any_candidate_flag": int(any_flag.sum()),
        "any_candidate_flag_pct_of_all_records": 100 * any_flag.mean(),
        "records_with_zero_candidate_flags": int(flag_count.eq(0).sum()),
        "records_with_one_candidate_flag": int(flag_count.eq(1).sum()),
        "records_with_two_candidate_flags": int(flag_count.eq(2).sum()),
        "records_with_three_candidate_flags": int(flag_count.eq(3).sum()),
        "records_removed": 0,
        "original_values_corrected": 0,
    }

    print(f"  Writing separate {name} output...", flush=True)
    df.to_csv(output_path, index=False)

    print(f"  Saved: {output_path.name}", flush=True)

    return field_rows, summary


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Check both outputs before starting either dataset.
    for config in CONFIGS:
        output = DATA / config["output"]
        if output.exists():
            raise FileExistsError(
                f"Output already exists: {output.name}. "
                "No existing flagged output will be overwritten."
            )

    field_rows = []
    summaries = []

    for config in CONFIGS:
        rows, summary = process_dataset(config)
        field_rows.extend(rows)
        summaries.append(summary)

    fields_report = pd.DataFrame(field_rows)
    summary_report = pd.DataFrame(summaries)

    fields_report.to_csv(
        REPORTS / "candidate_iqr_field_counts_week7.csv",
        index=False,
    )
    summary_report.to_csv(
        REPORTS / "candidate_iqr_run_summary_week7.csv",
        index=False,
    )

    print("\nPER-FIELD CANDIDATE COUNTS")
    print(fields_report.to_string(index=False))

    print("\nUNIQUE-RECORD SUMMARY")
    preview_columns = [
        "dataset",
        "records_read",
        "records_written",
        "all_three_fields_eligible",
        "at_least_one_field_ineligible",
        "unique_records_with_any_candidate_flag",
        "any_candidate_flag_pct_of_all_records",
        "records_with_zero_candidate_flags",
        "records_with_one_candidate_flag",
        "records_with_two_candidate_flags",
        "records_with_three_candidate_flags",
    ]
    print(summary_report[preview_columns].to_string(index=False))

    print("\nCANDIDATE FLAGGING COMPLETE")
    print("Separate Sold and Listing outputs were saved.")
    print("Original input files were not overwritten.")
    print("No rows were removed or original values corrected.")
    print("Per-field candidate flags are blank for ineligible values.")
    print(
        "The any-field flag means at least one eligible field "
        "exceeded its candidate bounds."
    )
    print(
        "A False any-field flag does not establish that every "
        "field was eligible or valid."
    )
    print("Candidate flags are not confirmed-error labels.")
    print("No exclusion rules were applied.")
    print("Keep full flagged datasets local; they contain source data.")


if __name__ == "__main__":
    main()