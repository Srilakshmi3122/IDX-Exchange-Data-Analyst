from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week7"

INPUT = DATA / "listings_residential_features_school_districts.csv"

FIELDS = ["ListPrice", "LivingArea", "DaysOnMarket"]

PERCENTILES = {
    "p01": 0.01,
    "p05": 0.05,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    REPORTS.mkdir(parents=True, exist_ok=True)

    print("1/4 Reading the three Listing numeric fields...", flush=True)

    df = pd.read_csv(
        INPUT,
        usecols=FIELDS,
        dtype="string",
        keep_default_na=False,
    )

    total = len(df)
    print(f"Records read: {total:,}", flush=True)

    if total == 0:
        raise ValueError("The input dataset contains no records.")

    print("2/4 Checking values and calculating summaries...", flush=True)

    quality_rows = []
    summary_rows = []
    iqr_rows = []
    eligible_masks = {}

    for field in FIELDS:
        text = df[field].str.strip()
        missing = text.eq("")

        numeric = pd.to_numeric(
            text.mask(missing),
            errors="coerce",
        ).astype("float64")

        parse_failure = ~missing & numeric.isna()
        finite = numeric.notna() & np.isfinite(numeric)
        nonfinite = numeric.notna() & ~np.isfinite(numeric)

        if field in ["ListPrice", "LivingArea"]:
            invalid = finite & numeric.le(0)
            rule = "Value must be greater than zero"
        else:
            invalid = finite & numeric.lt(0)
            rule = "Value must be zero or greater"

        eligible = finite & ~invalid
        eligible_masks[field] = eligible
        values = numeric.loc[eligible]

        category_total = (
            int(missing.sum())
            + int(parse_failure.sum())
            + int(nonfinite.sum())
            + int(invalid.sum())
            + int(eligible.sum())
        )

        if category_total != total:
            raise ValueError(
                f"Quality categories do not total all rows: {field}"
            )

        quality_rows.append({
            "field": field,
            "total_records": total,
            "missing_blank": int(missing.sum()),
            "numeric_parse_failures": int(parse_failure.sum()),
            "nonfinite_values": int(nonfinite.sum()),
            "business_rule_invalid": int(invalid.sum()),
            "eligible_for_iqr": int(eligible.sum()),
            "business_rule": rule,
        })

        if values.empty:
            raise ValueError(f"No eligible numeric values for {field}")

        summary = {
            "field": field,
            "eligible_records": len(values),
            "min": values.min(),
            "mean": values.mean(),
            "max": values.max(),
        }

        for label, quantile in PERCENTILES.items():
            summary[label] = values.quantile(quantile)

        summary_rows.append(summary)

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        below = values.lt(lower)
        above = values.gt(upper)
        outside = below | above

        iqr_rows.append({
            "field": field,
            "eligible_records": len(values),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "candidate_lower_bound": lower,
            "candidate_upper_bound": upper,
            "eligible_below_lower_bound": int(below.sum()),
            "eligible_above_upper_bound": int(above.sum()),
            "eligible_outside_bounds": int(outside.sum()),
            "eligible_outside_pct": 100 * outside.mean(),
            "iqr_zero": bool(iqr == 0),
        })

        print(f"  Profiled {field}", flush=True)

    quality = pd.DataFrame(quality_rows)
    distribution = pd.DataFrame(summary_rows)
    thresholds = pd.DataFrame(iqr_rows)

    all_eligible = pd.DataFrame(eligible_masks).all(axis=1)

    print("3/4 Saving aggregate reports...", flush=True)

    quality.to_csv(
        REPORTS / "listings_outlier_input_quality.csv",
        index=False,
    )

    distribution.to_csv(
        REPORTS / "listings_outlier_percentiles.csv",
        index=False,
    )

    thresholds.to_csv(
        REPORTS / "listings_outlier_candidate_iqr.csv",
        index=False,
    )

    pd.DataFrame([{
        "dataset": "listings",
        "input_file": INPUT.name,
        "records_read": total,
        "fields_profiled": len(FIELDS),
        "records_eligible_for_all_three_fields": int(
            all_eligible.sum()
        ),
        "records_not_eligible_for_at_least_one_field": int(
            (~all_eligible).sum()
        ),
        "iqr_multiplier": 1.5,
        "threshold_scope": (
            "Global Listing dataset, separately by field"
        ),
        "threshold_population": (
            "Finite values passing field business rule"
        ),
        "records_filtered": 0,
        "input_overwritten": False,
    }]).to_csv(
        REPORTS / "listings_outlier_profile_run_summary.csv",
        index=False,
    )

    print("4/4 Profiling complete.", flush=True)

    print("\nINPUT QUALITY")
    print(quality.to_string(index=False))

    print("\nPERCENTILES: ELIGIBLE VALUES ONLY")
    print(distribution.to_string(index=False))

    print("\nCANDIDATE IQR THRESHOLDS")
    print(thresholds.to_string(index=False))

    print("\nLISTING OUTLIER PROFILING COMPLETE")
    print(f"Records read: {total:,}")
    print(
        "Records eligible for all three fields: "
        f"{int(all_eligible.sum()):,}"
    )
    print("No records were filtered.")
    print("No dataset values were changed.")
    print("Candidate thresholds require review before filtering.")
    print("Outlier counts are per field and may overlap.")
    print("Missing values are not counted as statistical outliers.")
    print("Valid extreme values are not automatically data errors.")
    print("All reports must remain local.")


if __name__ == "__main__":
    main()