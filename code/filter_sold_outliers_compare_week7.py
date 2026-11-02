from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week7"

INPUT = DATA / (
    "sold_residential_features_school_districts"
    "_candidate_outliers_week7.csv"
)

OUTPUT = DATA / (
    "sold_residential_features_school_districts"
    "_iqr_filtered_week7.csv"
)

FIELDS = ["ClosePrice", "LivingArea", "DaysOnMarket"]
EXPECTED_RECORDS = 414054


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(series):
    text = series.astype("string").str.strip()
    values = pd.to_numeric(
        text.mask(text.eq("")),
        errors="coerce",
    ).astype("float64")
    return values.where(np.isfinite(values))


def boolean_values(series, label):
    require(
        series.isin(["True", "False"]).all(),
        f"Invalid Boolean text in {label}",
    )
    return series.eq("True").astype(bool)


def main():
    require(INPUT.is_file(), f"Missing input: {INPUT.name}")
    require(
        not OUTPUT.exists(),
        f"Output already exists: {OUTPUT.name}. "
        "Review it before rerunning; it will not be overwritten.",
    )

    REPORTS.mkdir(parents=True, exist_ok=True)

    threshold_path = REPORTS / "sold_outlier_candidate_iqr.csv"
    require(
        threshold_path.is_file(),
        f"Missing thresholds: {threshold_path.name}",
    )

    thresholds = pd.read_csv(threshold_path).set_index("field")
    require(
        thresholds.index.is_unique,
        "Duplicate fields in threshold report.",
    )

    print("1/5 Reading the full flagged Sold dataset...", flush=True)

    df = pd.read_csv(
        INPUT,
        dtype="string",
        keep_default_na=False,
    )

    total = len(df)
    require(
        total == EXPECTED_RECORDS,
        f"Expected {EXPECTED_RECORDS:,} records, found {total:,}.",
    )

    print("2/5 Checking eligibility and IQR flags...", flush=True)

    values_by_field = {}
    eligible_by_field = {}
    flagged_by_field = {}
    bounds_by_field = {}

    for field in FIELDS:
        require(field in thresholds.index, f"No bounds for {field}")

        values = numeric(df[field])
        values_by_field[field] = values

        if field == "DaysOnMarket":
            eligible = values.notna() & values.ge(0)
        else:
            eligible = values.notna() & values.gt(0)

        prefix = f"w7_{field}_"
        saved_eligible = boolean_values(
            df[prefix + "iqr_eligible"],
            f"{field}/eligible",
        )

        require(
            eligible.astype(bool).equals(saved_eligible),
            f"{field}: saved eligibility differs from current values.",
        )

        population = values.loc[eligible]
        require(not population.empty, f"No eligible values: {field}")

        q1 = population.quantile(0.25)
        q3 = population.quantile(0.75)
        iqr = q3 - q1

        lower = float(
            thresholds.loc[field, "candidate_lower_bound"]
        )
        upper = float(
            thresholds.loc[field, "candidate_upper_bound"]
        )

        require(
            np.isfinite(lower) and np.isfinite(upper),
            f"{field}: nonfinite IQR bounds.",
        )

        require(
            np.isclose(
                lower, q1 - 1.5 * iqr,
                rtol=1e-12, atol=1e-9,
            )
            and np.isclose(
                upper, q3 + 1.5 * iqr,
                rtol=1e-12, atol=1e-9,
            ),
            f"{field}: saved bounds differ from current quartiles.",
        )

        outside = eligible & (
            values.lt(lower) | values.gt(upper)
        )

        saved_candidate = df[
            prefix + "candidate_iqr_outlier"
        ]

        require(
            saved_candidate.loc[eligible].isin(
                ["True", "False"]
            ).all(),
            f"{field}: invalid candidate flags for eligible values.",
        )
        require(
            saved_candidate.loc[~eligible].eq("").all(),
            f"{field}: ineligible candidate flags must be blank.",
        )
        require(
            outside.astype(bool).equals(
                saved_candidate.eq("True").astype(bool)
            ),
            f"{field}: saved candidate flags differ from bounds.",
        )

        eligible_by_field[field] = eligible
        flagged_by_field[field] = outside
        bounds_by_field[field] = (lower, upper)

        print(f"  Checked {field}", flush=True)

    all_eligible = pd.DataFrame(
        eligible_by_field
    ).all(axis=1)

    any_flag = pd.DataFrame(
        flagged_by_field
    ).any(axis=1)

    saved_all = boolean_values(
        df["w7_all_three_fields_iqr_eligible"],
        "all-three eligibility",
    )
    saved_any = boolean_values(
        df["w7_any_field_candidate_iqr_outlier"],
        "any-field candidate flag",
    )

    require(
        all_eligible.astype(bool).equals(saved_all),
        "Combined eligibility flag mismatch.",
    )
    require(
        any_flag.astype(bool).equals(saved_any),
        "Combined candidate flag mismatch.",
    )

    excluded_ineligible = ~all_eligible
    excluded_iqr = all_eligible & any_flag
    keep = all_eligible & ~any_flag

    require(
        int(excluded_ineligible.sum())
        + int(excluded_iqr.sum())
        + int(keep.sum())
        == total,
        "Exclusion categories do not reconcile.",
    )
    require(keep.any(), "No records remain after filtering.")

    print("3/5 Calculating count and median comparisons...", flush=True)

    populations = [
        ("full_before_filtering", pd.Series(True, index=df.index)),
        ("all_three_eligible_before_iqr", all_eligible),
        ("filtered_after_iqr", keep),
    ]

    size_rows = []
    median_rows = []

    for label, mask in populations:
        size_rows.append({
            "population": label,
            "records": int(mask.sum()),
            "pct_of_full_input": 100 * mask.mean(),
        })

        for field in FIELDS:
            values = values_by_field[field].loc[mask]
            eligible_values = values_by_field[field].loc[
                mask & eligible_by_field[field]
            ]

            median_rows.append({
                "population": label,
                "field": field,
                "population_records": int(mask.sum()),
                "finite_numeric_values": int(values.notna().sum()),
                "median_finite_numeric_values": values.median(),
                "field_eligible_values": len(eligible_values),
                "median_field_eligible_values": eligible_values.median(),
            })

    sizes = pd.DataFrame(size_rows)
    medians = pd.DataFrame(median_rows)

    exclusions = pd.DataFrame([
        {
            "category": "excluded_ineligible_for_at_least_one_field",
            "records": int(excluded_ineligible.sum()),
            "pct_of_full_input": 100 * excluded_ineligible.mean(),
        },
        {
            "category": "excluded_iqr_among_all_three_eligible",
            "records": int(excluded_iqr.sum()),
            "pct_of_full_input": 100 * excluded_iqr.mean(),
        },
        {
            "category": "retained_in_filtered_dataset",
            "records": int(keep.sum()),
            "pct_of_full_input": 100 * keep.mean(),
        },
    ])

    print("4/5 Saving the separate filtered dataset...", flush=True)

    filtered = df.loc[keep].copy()

    for field in FIELDS:
        lower, upper = bounds_by_field[field]
        retained_values = values_by_field[field].loc[keep]

        require(
            retained_values.between(
                lower, upper, inclusive="both"
            ).all(),
            f"{field}: retained values fall outside IQR bounds.",
        )

    filtered.to_csv(OUTPUT, index=False)

    sizes.to_csv(
        REPORTS / "sold_iqr_filter_population_sizes_week7.csv",
        index=False,
    )
    medians.to_csv(
        REPORTS / "sold_iqr_filter_median_comparison_week7.csv",
        index=False,
    )
    exclusions.to_csv(
        REPORTS / "sold_iqr_filter_exclusion_breakdown_week7.csv",
        index=False,
    )

    print("5/5 Saving the written comparison...", flush=True)

    lines = [
        "WEEK 7 — SOLD IQR FILTERING COMPARISON",
        "",
        f"Full flagged input: {INPUT.name}",
        f"Separate filtered output: {OUTPUT.name}",
        "",
        "FILTER RULE",
        "Retain records eligible for ClosePrice, LivingArea, and",
        "DaysOnMarket and within each field's global 1.5-IQR bounds.",
        "Bounds are inclusive.",
        "Eligibility requires finite numeric values, positive price",
        "and area, and nonnegative DaysOnMarket.",
        "",
        "BOUNDS",
    ]

    for field, (lower, upper) in bounds_by_field.items():
        lines.append(f"{field}: {lower:,.2f} to {upper:,.2f}")

    lines.extend([
        "",
        "DATASET SIZE",
        f"Full input: {total:,}",
        f"All three fields eligible: {int(all_eligible.sum()):,}",
        (
            "Excluded for ineligibility: "
            f"{int(excluded_ineligible.sum()):,}"
        ),
        (
            "Excluded by IQR among all-three-eligible records: "
            f"{int(excluded_iqr.sum()):,}"
        ),
        f"Filtered output: {len(filtered):,}",
        f"Total excluded: {total - len(filtered):,}",
        f"Percentage retained: {100 * keep.mean():.2f}%",
        "",
        "Exclusion categories are mutually exclusive.",
        "Ineligible records are assigned to the ineligibility",
        "category even if another field has a candidate IQR flag.",
        "",
        "MEDIAN COMPARISON",
        "Full-input medians below use finite numeric values per field.",
        "The eligible baseline uses records eligible in all three fields.",
    ])

    for field in FIELDS:
        values = values_by_field[field]
        before = values.median()
        eligible_before = values.loc[all_eligible].median()
        after = values.loc[keep].median()

        lines.extend([
            "",
            field,
            f"  Full-input median: {before:,.2f}",
            f"  All-three-eligible baseline median: {eligible_before:,.2f}",
            f"  Filtered median: {after:,.2f}",
            f"  Change from full-input median: {after - before:,.2f}",
            (
                "  Change from eligible baseline median: "
                f"{after - eligible_before:,.2f}"
            ),
        ])

    lines.extend([
        "",
        "INTERPRETATION AND LIMITATIONS",
        "The filtered output represents the stated eligibility and",
        "global IQR rules, not independently verified error-free data.",
        "IQR exclusions may include valid extreme properties.",
        "The negative lower bounds do not flag suspicious positive",
        "low-end prices and areas documented in the review findings.",
        "No arbitrary low-end cutoffs or replacement values were added.",
        "Observed DOM/date discrepancies were not corrected.",
        "Geographic exclusions were not added.",
        "Median changes describe this filter's effect; they do not",
        "prove improved accuracy or representativeness.",
        "",
        "PRESERVATION",
        "The full flagged dataset was not modified.",
        "Original source datasets were not overwritten.",
        "Original values were not corrected.",
        "Retained rows preserve input order and all flagged columns.",
        "Keep the full and filtered datasets local.",
    ])

    written = "\n".join(lines) + "\n"

    (REPORTS / "sold_iqr_filter_written_comparison_week7.txt").write_text(
        written,
        encoding="utf-8",
    )

    print("\nPOPULATION SIZES")
    print(sizes.to_string(index=False))

    print("\nEXCLUSION BREAKDOWN")
    print(exclusions.to_string(index=False))

    print("\nMEDIAN COMPARISON")
    print(medians.to_string(index=False))

    print("\nSOLD IQR FILTERING COMPLETE")
    print(f"Saved filtered dataset: {OUTPUT.name}")
    print("Saved three aggregate CSV reports and a written comparison.")
    print("The full flagged dataset was preserved.")
    print("No original values were corrected.")
    print("Saved output still requires independent read-back verification.")
    print("Keep full and filtered datasets local.")


if __name__ == "__main__":
    main()