from pathlib import Path
import pandas as pd

# Project folders
PROJECT = Path.home() / "OneDrive" / "Desktop" / "IDX Exchange"
LISTED_FOLDER = PROJECT / "data_raw" / "listed"
SOLD_FOLDER = PROJECT / "data_raw" / "sold"
OUTPUT_FOLDER = PROJECT / "data_processed"

# Create the output folder if it does not already exist
OUTPUT_FOLDER.mkdir(exist_ok=True)

def combine_and_filter(input_folder, output_name, dataset_label):
    csv_files = sorted(input_folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    print(f"\n--- {dataset_label} dataset ---")
    print(f"Monthly files found: {len(csv_files)}")

    monthly_frames = []

    for file in csv_files:
        print(f"Reading: {file.name}")
        monthly_frames.append(pd.read_csv(file, low_memory=False))

    rows_before_concatenation = sum(len(frame) for frame in monthly_frames)
    combined = pd.concat(monthly_frames, ignore_index=True, sort=False)

    print(f"Rows before concatenation: {rows_before_concatenation:,}")
    print(f"Rows after concatenation:  {len(combined):,}")

    if "PropertyType" not in combined.columns:
        raise KeyError(
            "Column 'PropertyType' was not found. "
            f"Available columns include: {list(combined.columns[:20])}"
        )

    rows_before_filter = len(combined)

    residential = combined[
        combined["PropertyType"].astype(str).str.strip().str.lower() == "residential"
    ].copy()

    print(f"Rows before Residential filter: {rows_before_filter:,}")
    print(f"Rows after Residential filter:  {len(residential):,}")

    output_path = OUTPUT_FOLDER / output_name
    residential.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

combine_and_filter(
    LISTED_FOLDER,
    "combined_listings_residential.csv",
    "Listings"
)

combine_and_filter(
    SOLD_FOLDER,
    "combined_sold_residential.csv",
    "Sold transactions"
)