import os
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets", "On time")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "merged_ontime_2025.csv")

# Redundant or uninformative columns to drop unconditionally:
#   - Duplicate airport ID variants (SeqID, CityMarketID, StateFips, StateName, Wac)
#     are all alternative keys for the same airport; we keep AirportID + IATA code.
#   - DOT_ID_Reporting_Airline is redundant with Reporting_Airline.
#   - Quarter is fully derivable from Month.
#   - Flights is always 1.
#   - DistanceGroup is a binned version of Distance (keep the raw value).
#   - WheelsOff / WheelsOn are not needed for rotation chain construction.
#   - DepartureDelayGroups / ArrivalDelayGroups are binned versions of the delay columns.
REDUNDANT_COLS = [
    "DOT_ID_Reporting_Airline",
    "OriginAirportSeqID", "OriginCityMarketID", "OriginStateFips",
    "OriginStateName", "OriginWac",
    "DestAirportSeqID", "DestCityMarketID", "DestStateFips",
    "DestStateName", "DestWac",
    "DepartureDelayGroups", "ArrivalDelayGroups",
    "Quarter",
    "Flights",
    "DistanceGroup",
    "WheelsOff", "WheelsOn",
    "DivAirportLandings",  # always 0 after dropping diverted flights
]

# Delay-cause columns are NULL when a flight has no reportable delay (not truly missing).
# Protect them from the null-rate drop and fill with 0 instead.
DELAY_CAUSE_COLS = [
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]

NULL_THRESHOLD = 0.50  # drop columns where more than 50% of rows are null


def load_and_merge() -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"Found {len(csv_files)} CSV files.\n")

    dfs = []
    for f in csv_files:
        print(f"  Loading {os.path.basename(f)} ...")
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged shape before cleaning : {merged.shape}")

    # --- Drop redundant columns ---
    drop_redundant = [c for c in REDUNDANT_COLS if c in merged.columns]
    merged.drop(columns=drop_redundant, inplace=True)
    print(f"Dropped {len(drop_redundant)} redundant columns.")

    # --- Drop high-null columns (protect delay-cause columns) ---
    null_rates = merged.isnull().mean()
    high_null = [
        c for c in null_rates[null_rates > NULL_THRESHOLD].index
        if c not in DELAY_CAUSE_COLS
    ]
    merged.drop(columns=high_null, inplace=True)
    print(f"Dropped {len(high_null)} high-null columns (>{NULL_THRESHOLD:.0%} null):")
    for c in high_null:
        print(f"    {c:35s}  ({null_rates[c]:.1%} null)")

    # --- Fill delay-cause NULLs with 0 ---
    for col in DELAY_CAUSE_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # --- Remove cancelled and diverted flights ---
    # These cannot form valid aircraft rotation chains and have missing timing data.
    n_before = len(merged)
    merged = merged[
        (merged["Cancelled"] == 0) & (merged["Diverted"] == 0)
    ].copy()
    print(f"\nRemoved {n_before - len(merged):,} cancelled / diverted rows.")

    # --- Parse date column ---
    merged["FlightDate"] = pd.to_datetime(merged["FlightDate"])

    print(f"\nFinal shape : {merged.shape}")
    print(f"Columns ({len(merged.columns)}):")
    for col in merged.columns:
        print(f"    {col}")

    return merged


if __name__ == "__main__":
    df = load_and_merge()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
