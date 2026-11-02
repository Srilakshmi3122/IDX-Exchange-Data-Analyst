from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week7"

FULL = DATA / (
    "sold_residential_features_school_districts"
    "_candidate_outliers_week7.csv"
)

FILTERED = DATA / (
    "sold_residential_features_school_districts"
    "_iqr_filtered_week7.csv"
)

FIELDS = ["ClosePrice", "LivingArea", "DaysOnMarket"]


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


def same_number(actual, expected, label):
    require(
        bool(np.isclose(
            float(actual),
            float(expected),
            rtol=1e-12,
            atol=1e-9,
            equal_nan=True,
        )),
        f"{label}: saved={actual}, expected={expected}",
    )


def main():
    paths = {
        "full": FULL,
        "filtered": FILTERED,
        "bounds": REPORTS / "sold_outlier_candidate_iqr.csv",
        "sizes": REPORTS / "sold_iqr_filter_population_sizes_week7.csv",
        "medians": REPORTS / "sold_iqr_filter_median_comparison_week7.csv",
        "exclusions": REPORTS / "sold_iqr_filter_exclusion_breakdown_week7.csv",
        "written": REPORTS / "sold_iqr_filter_written_comparison_week7.txt",
    }

    for label, path in paths.items():
        require(path.is_file(), f"Missing {label}: {path.name}")
        require(path.stat().st_size > 0, f"Empty file: {path.name}")

    print("1/6 Reading saved full and filtered datasets...", flush=True)

    full = pd.read_csv(
        FULL,
        dtype="string",
        keep_default_na=False,
    )
    filtered = pd.read_csv(
        FILTERED,
        dtype="string",
        keep_default_na=False,
    )

    require(len(full) == 414054, "Unexpected full dataset count.")
    require(len(filtered) == 348780, "Unexpected filtered dataset count.")
    require(
        full.columns.equals(filtered.columns),
        "Filtered columns or column order differ from full input.",
    )

    print("2/6 Independently reconstructing IQR rules...", flush=True)

    bounds = pd.read_csv(paths["bounds"]).set_index("field")
    require(bounds.index.is_unique, "Duplicate threshold fields.")

    values_by_field = {}
    eligible_by_field = {}
    outside_by_field = {}

    for field in FIELDS:
        values = numeric(full[field])
        eligible = values.notna()

        if field == "DaysOnMarket":
            eligible &= values.ge(0)
        else:
            eligible &= values.gt(0)

        population = values.loc[eligible]
        require(not population.empty, f"No eligible values: {field}")

        q1 = population.quantile(0.25)
        q3 = population.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        require(field in bounds.index, f"Missing bounds: {field}")
        same_number(
            bounds.loc[field, "candidate_lower_bound"],
            lower,
            f"{field} lower bound",
        )
        same_number(
            bounds.loc[field, "candidate_upper_bound"],
            upper,
            f"{field} upper bound",
        )

        outside = eligible & (
            values.lt(lower) | values.gt(upper)
        )

        prefix = f"w7_{field}_"
        saved_eligible = full[prefix + "iqr_eligible"]
        saved_candidate = full[prefix + "candidate_iqr_outlier"]

        require(
            saved_eligible.isin(["True", "False"]).all(),
            f"{field}: invalid eligibility text.",
        )
        require(
            np.array_equal(
                saved_eligible.eq("True").to_numpy(dtype=bool),
                eligible.to_numpy(dtype=bool),
            ),
            f"{field}: eligibility flag mismatch.",
        )
        require(
            saved_candidate.loc[eligible].isin(
                ["True", "False"]
            ).all(),
            f"{field}: invalid eligible candidate flag.",
        )
        require(
            saved_candidate.loc[~eligible].eq("").all(),
            f"{field}: ineligible candidate flag is not blank.",
        )
        require(
            np.array_equal(
                saved_candidate.eq("True").to_numpy(dtype=bool),
                outside.to_numpy(dtype=bool),
            ),
            f"{field}: candidate flag mismatch.",
        )

        values_by_field[field] = values
        eligible_by_field[field] = eligible
        outside_by_field[field] = outside

        print(f"  Verified {field}", flush=True)

    all_eligible = pd.DataFrame(eligible_by_field).all(axis=1)
    any_outside = pd.DataFrame(outside_by_field).any(axis=1)

    for column, expected in [
        ("w7_all_three_fields_iqr_eligible", all_eligible),
        ("w7_any_field_candidate_iqr_outlier", any_outside),
    ]:
        saved = full[column]
        require(
            saved.isin(["True", "False"]).all(),
            f"Invalid combined flag: {column}",
        )
        require(
            np.array_equal(
                saved.eq("True").to_numpy(dtype=bool),
                expected.to_numpy(dtype=bool),
            ),
            f"Combined flag mismatch: {column}",
        )

    keep = all_eligible & ~any_outside
    ineligible = ~all_eligible
    excluded_iqr = all_eligible & any_outside

    require(int(all_eligible.sum()) == 413619, "Eligible count mismatch.")
    require(int(ineligible.sum()) == 435, "Ineligible count mismatch.")
    require(int(excluded_iqr.sum()) == 64839, "IQR exclusion count mismatch.")
    require(int(keep.sum()) == len(filtered), "Retained count mismatch.")
    require(
        int(ineligible.sum()) + int(excluded_iqr.sum()) + int(keep.sum())
        == len(full),
        "Exclusion categories do not reconcile.",
    )

    print("3/6 Comparing every retained row and column...", flush=True)

    expected = full.loc[keep].reset_index(drop=True)

    require(
        expected.shape == filtered.shape,
        "Expected and saved filtered shapes differ.",
    )

    for column in full.columns:
        require(
            expected[column].equals(filtered[column]),
            f"Saved values or row order differ in column: {column}",
        )

    del expected

    for field in FIELDS:
        saved_values = numeric(filtered[field])
        require(saved_values.notna().all(), f"Missing retained {field}")

        if field == "DaysOnMarket":
            require(saved_values.ge(0).all(), "Negative retained DOM.")
        else:
            require(saved_values.gt(0).all(), f"Nonpositive {field}.")

        eligible_values = values_by_field[field].loc[
            eligible_by_field[field]
        ]
        q1 = eligible_values.quantile(0.25)
        q3 = eligible_values.quantile(0.75)
        iqr = q3 - q1

        require(
            saved_values.between(
                q1 - 1.5 * iqr,
                q3 + 1.5 * iqr,
                inclusive="both",
            ).all(),
            f"Retained {field} outside reconstructed bounds.",
        )

    print("4/6 Checking saved count and exclusion reports...", flush=True)

    populations = {
        "full_before_filtering": pd.Series(True, index=full.index),
        "all_three_eligible_before_iqr": all_eligible,
        "filtered_after_iqr": keep,
    }

    sizes = pd.read_csv(paths["sizes"]).set_index("population")
    require(sizes.index.is_unique, "Duplicate population report rows.")
    require(
        set(sizes.index) == set(populations),
        "Unexpected population report labels.",
    )

    for label, mask in populations.items():
        same_number(sizes.loc[label, "records"], mask.sum(), label)
        same_number(
            sizes.loc[label, "pct_of_full_input"],
            100 * mask.mean(),
            f"{label} percentage",
        )

    categories = {
        "excluded_ineligible_for_at_least_one_field": ineligible,
        "excluded_iqr_among_all_three_eligible": excluded_iqr,
        "retained_in_filtered_dataset": keep,
    }

    exclusions = pd.read_csv(paths["exclusions"]).set_index("category")
    require(exclusions.index.is_unique, "Duplicate exclusion report rows.")
    require(
        set(exclusions.index) == set(categories),
        "Unexpected exclusion report labels.",
    )

    for label, mask in categories.items():
        same_number(exclusions.loc[label, "records"], mask.sum(), label)
        same_number(
            exclusions.loc[label, "pct_of_full_input"],
            100 * mask.mean(),
            f"{label} percentage",
        )

    print("5/6 Checking all saved median report values...", flush=True)

    medians = pd.read_csv(paths["medians"]).set_index(
        ["population", "field"]
    )
    require(medians.index.is_unique, "Duplicate median report rows.")

    expected_keys = {
        (label, field)
        for label in populations
        for field in FIELDS
    }
    require(
        set(medians.index) == expected_keys,
        "Unexpected median report rows.",
    )

    for label, mask in populations.items():
        for field in FIELDS:
            row = medians.loc[(label, field)]
            values = values_by_field[field].loc[mask]
            eligible_values = values_by_field[field].loc[
                mask & eligible_by_field[field]
            ]

            checks = {
                "population_records": mask.sum(),
                "finite_numeric_values": values.notna().sum(),
                "median_finite_numeric_values": values.median(),
                "field_eligible_values": len(eligible_values),
                "median_field_eligible_values": eligible_values.median(),
            }

            for column, expected_value in checks.items():
                same_number(
                    row[column],
                    expected_value,
                    f"{label}/{field}/{column}",
                )

            if label == "filtered_after_iqr":
                same_number(
                    row["median_finite_numeric_values"],
                    numeric(filtered[field]).median(),
                    f"Saved filtered CSV median: {field}",
                )

    print("6/6 Checking written report presence...", flush=True)

    written = paths["written"].read_text(encoding="utf-8")
    require(
        "WEEK 7" in written
        and "MEDIAN COMPARISON" in written
        and "INTERPRETATION AND LIMITATIONS" in written,
        "Written report is missing expected sections.",
    )

    print("\nINDEPENDENT READ-BACK VERIFICATION PASSED")
    print(f"Full flagged records: {len(full):,}")
    print(f"All-three-eligible records: {int(all_eligible.sum()):,}")
    print(f"Ineligible exclusions: {int(ineligible.sum()):,}")
    print(f"IQR exclusions among eligible records: {int(excluded_iqr.sum()):,}")
    print(f"Verified filtered records: {len(filtered):,}")
    print(f"Verified columns: {len(filtered.columns):,}")
    print("Every saved filtered value matches its expected source row.")
    print("Retained row order and all columns match.")
    print("Eligibility, IQR bounds, and flags reconcile.")
    print("Count, percentage, and median CSV reports reconcile.")
    print("Written report exists and contains expected sections.")
    print("No datasets or reports were modified by this verifier.")


if __name__ == "__main__":
    main()