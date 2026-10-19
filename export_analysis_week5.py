from pathlib import Path
import gc
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week5"
REPORTS.mkdir(parents=True, exist_ok=True)

CORE_COLUMNS = """
ListingKey ListingKeyNumeric ListingId
PropertyType PropertySubType MlsStatus
CloseDate PurchaseContractDate ListingContractDate ContractStatusChangeDate
ClosePrice ListPrice OriginalListPrice
LivingArea LotSizeAcres BedroomsTotal BathroomsTotalInteger
DaysOnMarket YearBuilt
Latitude Longitude latfilled lonfilled
UnparsedAddress City CountyOrParish StateOrProvince PostalCode MLSAreaMajor
ListAgentFullName ListAgentFirstName ListAgentLastName
CoListAgentFirstName CoListAgentLastName
BuyerAgentMlsId BuyerAgentFirstName BuyerAgentLastName
CoBuyerAgentFirstName
ListOfficeName BuyerOfficeName CoListOfficeName
ListAgentAOR BuyerAgentAOR BuyerOfficeAOR
ElementarySchool MiddleOrJuniorSchool HighSchool
ElementarySchoolDistrict MiddleOrJuniorSchoolDistrict HighSchoolDistrict
OriginatingSystemName OriginatingSystemSubName
year_month rate_30yr_fixed
""".split()

OPTIONAL_COLUMNS = {
    "latfilled",
    "lonfilled",
    "ListAgentAOR",
    "BuyerAgentAOR",
    "OriginatingSystemName",
    "OriginatingSystemSubName",
}

summary_rows = []
selection_rows = []

for label in ["listings", "sold"]:
    print(f"\n--- {label.upper()} ---", flush=True)

    input_path = DATA / f"{label}_residential_prepared.csv"
    output_path = DATA / f"{label}_residential_analysis.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    columns = pd.read_csv(input_path, nrows=0).columns.tolist()

    missing_required = [
        c for c in CORE_COLUMNS
        if c not in columns and c not in OPTIONAL_COLUMNS
    ]

    if missing_required:
        raise ValueError(
            f"{label}: required columns missing: {missing_required}"
        )

    missing_optional = [
        c for c in CORE_COLUMNS
        if c in OPTIONAL_COLUMNS and c not in columns
    ]

    if missing_optional:
        print(
            "Optional fields absent; skipping: "
            + ", ".join(missing_optional),
            flush=True,
        )

    flag_columns = [
        c for c in columns
        if c.endswith("_flag")
    ]

    if not flag_columns:
        raise ValueError(f"{label}: no quality flags found.")

    keep = [
        c for c in CORE_COLUMNS
        if c in columns
    ] + flag_columns

    # Select columns without reinterpreting stored CSV values.
    df = pd.read_csv(
        input_path,
        dtype="string",
        keep_default_na=False,
    )

    rows_before = len(df)
    reduced = df.loc[:, keep]

    if len(reduced) != rows_before:
        raise ValueError("Unexpected row-count change.")

    if not reduced[flag_columns].equals(df[flag_columns]):
        raise ValueError("Quality flags changed unexpectedly.")

    if not reduced["rate_30yr_fixed"].equals(df["rate_30yr_fixed"]):
        raise ValueError("Mortgage-rate values changed unexpectedly.")

    for column in columns:
        retained = column in keep

        if retained:
            if column.endswith("_flag"):
                reason = "Retained quality flag"
            else:
                reason = "Retained analysis or context field"
        elif column.endswith("_raw"):
            reason = "Backup retained in full prepared and review files"
        else:
            reason = "Outside selected scope; retained in full files"

        selection_rows.append({
            "dataset": label,
            "column": column,
            "retained_in_analysis_export": retained,
            "reason": reason,
        })

    # Document selected optional fields absent from this dataset.
    for column in missing_optional:
        selection_rows.append({
            "dataset": label,
            "column": column,
            "retained_in_analysis_export": False,
            "reason": "Optional field absent from source dataset",
        })

    print("Saving reduced analysis export...", flush=True)
    reduced.to_csv(output_path, index=False)

    summary_rows.append({
        "dataset": label,
        "rows_before": rows_before,
        "rows_after": len(reduced),
        "columns_before": len(columns),
        "columns_after": len(keep),
        "columns_excluded": len(columns) - len(keep),
        "quality_flags_retained": len(flag_columns),
        "optional_fields_absent": len(missing_optional),
        "output_file": output_path.name,
    })

    print(f"Saved: {output_path.name}", flush=True)

    del df, reduced
    gc.collect()

pd.DataFrame(selection_rows).to_csv(
    REPORTS / "column_selection_report.csv",
    index=False,
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    REPORTS / "analysis_export_summary.csv",
    index=False,
)

print("\nANALYSIS EXPORT COMPLETE")
print(summary.to_string(index=False))
print("\nFull prepared and review files were not overwritten.")
print("This step selected columns only; no rows or values were changed.")