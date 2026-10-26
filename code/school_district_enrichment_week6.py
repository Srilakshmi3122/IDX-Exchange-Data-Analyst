from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data_processed"
REPORTS = PROJECT / "reports" / "week6"
BOUNDARY = (
    PROJECT / "data_external" / "school_districts"
    / "DistrictAreas2425.shp"
)

DISTRICT_TYPES = {
    "Elementary": "elementary",
    "Unified": "unified",
    "High": "high",
}

DATASETS = {
    "sold": "sold_residential_features.csv",
    "listings": "listings_residential_features.csv",
}

CHUNK_SIZE = 50000


def load_boundaries():
    print("Reading school-district boundaries...", flush=True)
    g = gpd.read_file(BOUNDARY)

    required = ["CDSCode", "DistrictNa", "DistrictTy", "Year"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"Missing boundary fields: {missing}")

    if g.crs is None:
        raise ValueError("Boundary coordinate system is missing.")

    if g.geometry.isna().any() or g.geometry.is_empty.any():
        raise ValueError("Missing or empty boundary geometry.")

    if g["CDSCode"].isna().any() or g["CDSCode"].duplicated().any():
        raise ValueError("Boundary district codes are missing or duplicated.")

    unknown = set(g["DistrictTy"]) - set(DISTRICT_TYPES)
    if unknown:
        raise ValueError(f"Unexpected district types: {unknown}")

    if set(g["Year"].astype(str)) != {"2024-25"}:
        raise ValueError("Unexpected boundary year.")

    bad = ~g.geometry.is_valid

    repair_report = g.loc[
        bad, ["CDSCode", "DistrictNa", "DistrictTy"]
    ].copy()
    repair_report["original_validity_reason"] = (
        g.loc[bad].geometry.is_valid_reason()
    )

    g.loc[bad, "geometry"] = g.loc[bad].geometry.make_valid()

    if (
        (~g.geometry.is_valid).any()
        or g.geometry.isna().any()
        or g.geometry.is_empty.any()
    ):
        raise ValueError("Boundary repair did not pass geometry checks.")

    if not g.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Non-polygon geometry remains after repair.")

    repair_report["repaired_geometry_type"] = (
        g.loc[bad].geometry.geom_type
    )
    repair_report.to_csv(
        REPORTS / "school_district_boundary_repairs.csv",
        index=False,
    )

    g["CDSCode"] = g["CDSCode"].astype("string")
    g = g[required + ["geometry"]].to_crs("EPSG:4326")

    print(
        f"Boundary records: {len(g):,}; "
        f"geometries repaired: {int(bad.sum())}",
        flush=True,
    )
    return g


def enrich_chunk(original, boundaries):
    df = original.copy()

    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")

    missing = (
        df["Latitude"].str.strip().eq("")
        | df["Longitude"].str.strip().eq("")
    )
    usable = (
        lat.between(-90, 90)
        & lon.between(-180, 180)
        & lat.ne(0)
        & lon.ne(0)
    )

    df["school_coordinate_status"] = "invalid_coordinate"
    df.loc[missing, "school_coordinate_status"] = "missing_coordinate"
    df.loc[usable, "school_coordinate_status"] = "usable_coordinate"

    # MLS Latitude/Longitude are interpreted as WGS84 decimal degrees.
    # Range checks do not establish coordinate accuracy.
    points = gpd.GeoDataFrame(
        index=df.index[usable],
        geometry=gpd.points_from_xy(
            lon.loc[usable],
            lat.loc[usable],
            crs="EPSG:4326",
        ),
        crs="EPSG:4326",
    )

    for district_type, prefix in DISTRICT_TYPES.items():
        stem = f"school_{prefix}"
        count_col = f"{stem}_match_count"
        status_col = f"{stem}_status"
        code_col = f"{stem}_cdscode"
        name_col = f"{stem}_name"

        df[count_col] = pd.Series(
            pd.NA, index=df.index, dtype="Int64"
        )
        df[status_col] = df["school_coordinate_status"]
        df[code_col] = pd.Series(
            pd.NA, index=df.index, dtype="string"
        )
        df[name_col] = pd.Series(
            pd.NA, index=df.index, dtype="string"
        )

        df.loc[usable, count_col] = 0
        df.loc[usable, status_col] = "no_match"

        if points.empty:
            continue

        polygons = boundaries.loc[
            boundaries["DistrictTy"].eq(district_type),
            ["CDSCode", "DistrictNa", "geometry"],
        ]

        # Strict interior matching; no nearest-district fallback.
        hits = gpd.sjoin(
            points,
            polygons,
            how="inner",
            predicate="within",
        )

        if hits.empty:
            continue

        counts = hits.groupby(level=0).size()
        df.loc[counts.index, count_col] = counts.astype("Int64")
        df.loc[counts.index, status_col] = "multiple_matches"

        unique_rows = counts.index[counts.eq(1)]
        single = hits.loc[hits.index.isin(unique_rows)]

        df.loc[unique_rows, status_col] = "matched"
        df.loc[single.index, code_col] = single["CDSCode"]
        df.loc[single.index, name_col] = single["DistrictNa"]

    df["school_boundary_year"] = "2024-25"
    df["school_match_method"] = "within"

    if len(df) != len(original):
        raise ValueError("Record count changed within a chunk.")

    if not df[original.columns].equals(original):
        raise ValueError("Original values or record order changed.")

    return df


def process_dataset(label, filename, boundaries):
    source = DATA / filename
    output = DATA / f"{source.stem}_school_districts.csv"
    partial = output.with_suffix(".csv.partial")

    if not source.exists():
        raise FileNotFoundError(source)

    if output.exists() or partial.exists():
        raise FileExistsError(
            f"Output or partial output already exists for {label}. "
            "Inspect existing files before rerunning."
        )

    columns = pd.read_csv(source, nrows=0).columns.tolist()

    for field in ["Latitude", "Longitude"]:
        if field not in columns:
            raise ValueError(f"{label}: missing {field}")

    if any(c.startswith("school_") for c in columns):
        raise ValueError(f"{label}: school-prefixed fields already exist.")

    total = 0
    status_totals = {}
    sample_parts = []
    sample_rows = 0

    print(f"\nProcessing {label.upper()}...", flush=True)

    reader = pd.read_csv(
        source,
        dtype="string",
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, original in enumerate(reader, start=1):
        enriched = enrich_chunk(original, boundaries)
        first = chunk_number == 1

        enriched.to_csv(
            partial,
            mode="w" if first else "a",
            header=first,
            index=False,
        )

        total += len(enriched)

        status_columns = [
            "school_coordinate_status",
            *[
                f"school_{prefix}_status"
                for prefix in DISTRICT_TYPES.values()
            ],
        ]

        for field in status_columns:
            counts = enriched[field].value_counts(dropna=False)
            for status, count in counts.items():
                key = (field, status)
                status_totals[key] = (
                    status_totals.get(key, 0) + int(count)
                )

        if sample_rows < 10:
            take = enriched.head(10 - sample_rows)
            sample_columns = [
                c for c in ["ListingKey", "Latitude", "Longitude"]
                if c in enriched.columns
            ] + [
                c for c in enriched.columns
                if c.startswith("school_")
            ]
            sample_parts.append(take[sample_columns])
            sample_rows += len(take)

        print(
            f"  Chunk {chunk_number}: {total:,} records written",
            flush=True,
        )

    if total == 0:
        raise ValueError(f"{label}: input contains no records.")

    saved_rows = sum(
        len(chunk)
        for chunk in pd.read_csv(
            partial,
            usecols=[0],
            dtype="string",
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        )
    )
    if saved_rows != total:
        raise ValueError("Saved output row count does not match.")

    coverage = pd.DataFrame([
        {
            "dataset": label,
            "status_field": field,
            "status": status,
            "record_count": count,
        }
        for (field, status), count in status_totals.items()
    ])

    for field, group in coverage.groupby("status_field"):
        if int(group["record_count"].sum()) != total:
            raise ValueError(f"Status count mismatch: {field}")

    partial.rename(output)

    coverage.to_csv(
        REPORTS / f"{label}_school_district_coverage.csv",
        index=False,
    )
    pd.concat(sample_parts, ignore_index=True).to_csv(
        REPORTS / f"{label}_school_district_sample.csv",
        index=False,
    )

    added = len(enriched.columns) - len(columns)

    pd.DataFrame([{
        "dataset": label,
        "records_processed": total,
        "saved_records": saved_rows,
        "columns_before": len(columns),
        "columns_after": len(enriched.columns),
        "columns_added": added,
        "original_values_and_order_check": "PASS",
        "saved_row_count_check": "PASS",
        "boundary_year": "2024-25",
        "point_crs_assumption": "EPSG:4326",
        "match_predicate": "within",
        "output_file": output.name,
    }]).to_csv(
        REPORTS / f"{label}_school_district_run_summary.csv",
        index=False,
    )

    print(f"\n{label.upper()} SCHOOL-DISTRICT COVERAGE")
    print(coverage.to_string(index=False))
    print(f"Saved {total:,} records to {output.name}", flush=True)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    boundaries = load_boundaries()

    for label, filename in DATASETS.items():
        process_dataset(label, filename, boundaries)

    print("\nSCHOOL-DISTRICT ENRICHMENT COMPLETE")
    print("Original feature datasets and boundary files were not overwritten.")
    print("Ambiguous matches were counted but not assigned a district.")
    print("No-match does not prove that a property has no school district.")
    print("2024-25 boundaries were used, not date-specific historical boundaries.")
    print("Assignments are geographic matches, not enrollment guarantees.")
    print("All enriched CSVs and property-level samples must remain local.")


if __name__ == "__main__":
    main()