"""
Stage 1 — Aircraft rotation chain construction and delay amplification factor (α) labelling.

For every consecutive pair of flights operated by the same tail number where
Dest[i] == Origin[i+1], we record the turnaround event and compute:

    α = max(0, outbound_delay − inbound_delay) / max(1, inbound_delay)

α > 0  means the airport amplified (worsened) the delay.
α = 0  means the airport absorbed all incoming delay.
α > 1  means the outbound delay exceeded double the inbound delay.

Actual datetimes are derived as:
    dep_datetime  = FlightDate + DepTime (HHMM)
    arr_datetime  = dep_datetime + ActualElapsedTime (minutes)

This avoids all midnight-crossing edge cases without relying on ArrTime directly.

Output: data/rotation_chains.csv
"""

import os
import pandas as pd
import numpy as np

INPUT_PATH = os.path.join(os.path.dirname(__file__), "merged_ontime_2025.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "rotation_chains.csv")

MIN_GROUND_MIN = 20     # discard sub-20-min turnarounds (data artifacts / pushback overlaps)
MAX_GROUND_MIN = 720    # discard > 12-hour gaps (overnight maintenance / aircraft swaps)


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """Convert a HHMM integer series to elapsed minutes since midnight."""
    s = series.fillna(0).astype(int)
    return (s // 100) * 60 + (s % 100)


def build_rotation_chains(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Build actual departure and arrival datetimes
    dep_min = hhmm_to_minutes(df["DepTime"])
    df["dep_datetime"] = pd.to_datetime(df["FlightDate"]) + pd.to_timedelta(dep_min, unit="min")
    df["arr_datetime"] = df["dep_datetime"] + pd.to_timedelta(
        df["ActualElapsedTime"].fillna(0), unit="min"
    )

    # Build scheduled departure and arrival datetimes
    sched_dep_min = hhmm_to_minutes(df["CRSDepTime"])
    df["sched_dep_datetime"] = pd.to_datetime(df["FlightDate"]) + pd.to_timedelta(
        sched_dep_min, unit="min"
    )
    df["sched_arr_datetime"] = df["sched_dep_datetime"] + pd.to_timedelta(
        df["CRSElapsedTime"].fillna(0), unit="min"
    )

    # Sort chronologically within each tail number
    df = df.sort_values(["Tail_Number", "dep_datetime"]).reset_index(drop=True)

    # Columns belonging to the inbound (current) and outbound (next) flight
    inbound_cols = [
        "Tail_Number", "FlightDate", "Reporting_Airline", "IATA_CODE_Reporting_Airline",
        "Flight_Number_Reporting_Airline",
        "Origin", "OriginAirportID", "Dest", "DestAirportID",
        "arr_datetime", "sched_arr_datetime",
        "ArrDelay", "DayOfWeek",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
        "Distance",
    ]
    outbound_cols = [
        "Tail_Number", "Origin", "OriginAirportID",
        "dep_datetime", "sched_dep_datetime",
        "Flight_Number_Reporting_Airline",
        "DepDelay", "DepDelayMinutes", "DepTimeBlk", "CRSDepTime",
    ]

    inb = df[inbound_cols].copy()
    out = df[outbound_cols].shift(-1).copy()
    out.columns = [f"next_{c}" for c in outbound_cols]

    combined = pd.concat([inb, out], axis=1)

    # Valid rotation: same tail number and aircraft stayed at the same airport
    mask = (
        (combined["Tail_Number"] == combined["next_Tail_Number"]) &
        (combined["Dest"] == combined["next_Origin"])
    )
    rot = combined[mask].copy()

    # Ground time
    rot["actual_ground_min"] = (
        (rot["next_dep_datetime"] - rot["arr_datetime"]).dt.total_seconds() / 60
    )
    rot["scheduled_ground_min"] = (
        (rot["next_sched_dep_datetime"] - rot["sched_arr_datetime"]).dt.total_seconds() / 60
    )

    # Filter implausible turnarounds
    rot = rot[
        (rot["actual_ground_min"] >= MIN_GROUND_MIN) &
        (rot["actual_ground_min"] <= MAX_GROUND_MIN)
    ].copy()

    # Delay amplification factor
    inbound_delay = rot["ArrDelay"].fillna(0)
    outbound_delay = rot["next_DepDelay"].fillna(0)
    rot["inbound_delay"] = inbound_delay
    rot["outbound_delay"] = outbound_delay
    rot["alpha"] = (
        np.maximum(0.0, outbound_delay - inbound_delay) /
        np.maximum(1.0, inbound_delay)
    )

    # Minimum buffer needed to reduce this event's α below 0.2:
    #   outbound_delay_buffered = outbound_delay - B
    #   α(B) < 0.2  →  B > outbound_delay − 1.2 × inbound_delay
    rot["B_min"] = np.maximum(
        0.0, outbound_delay - 1.2 * inbound_delay
    )

    rot = rot.rename(columns={
        "Dest": "turnaround_airport",
        "DestAirportID": "turnaround_airport_id",
        "Flight_Number_Reporting_Airline": "inbound_flight",
        "next_Flight_Number_Reporting_Airline": "outbound_flight",
        "next_dep_datetime": "outbound_dep_datetime",
        "next_DepDelay": "_raw",
        "next_DepDelayMinutes": "outbound_delay_minutes",
        "next_DepTimeBlk": "dep_time_blk",
        "next_CRSDepTime": "sched_dep_time",
    })

    output_cols = [
        "Tail_Number", "FlightDate", "Reporting_Airline", "IATA_CODE_Reporting_Airline",
        "turnaround_airport", "turnaround_airport_id",
        "inbound_flight", "outbound_flight",
        "arr_datetime", "outbound_dep_datetime",
        "actual_ground_min", "scheduled_ground_min",
        "inbound_delay", "outbound_delay", "outbound_delay_minutes",
        "alpha", "B_min",
        "DayOfWeek", "dep_time_blk", "sched_dep_time",
        "Distance",
        "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    ]
    return rot[output_cols].reset_index(drop=True)


if __name__ == "__main__":
    print("Loading merged On-Time data...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} flights")

    n_before = len(df)
    df = df.dropna(subset=["Tail_Number", "DepTime", "ActualElapsedTime", "CRSElapsedTime"])
    print(f"  Dropped {n_before - len(df):,} rows with missing tail / timing fields")

    print("\nBuilding rotation chains...")
    chains = build_rotation_chains(df)

    print(f"\n  Rotation events   : {len(chains):,}")
    print(f"  Airports covered  : {chains['turnaround_airport'].nunique()}")
    print(f"\n  alpha statistics")
    print(chains["alpha"].describe().to_string())
    print(f"\n  High-amplification events")
    print(f"    alpha > 0.5 : {(chains['alpha'] > 0.5).sum():,}  ({(chains['alpha'] > 0.5).mean():.1%})")
    print(f"    alpha > 1.0 : {(chains['alpha'] > 1.0).sum():,}  ({(chains['alpha'] > 1.0).mean():.1%})")

    chains.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
