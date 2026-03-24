"""
Assemble the temporal node state matrix from rotation chain α labels.

For each (turnaround_airport, FlightDate, dep_time_blk) cell we compute:
    mean_alpha          — average delay amplification (the primary prediction target)
    median_alpha        — robust central tendency
    p75_alpha           — 75th percentile (high-amplification risk)
    mean_inbound_delay  — average inbound delay (d_i(t) in the GNN formulation)
    mean_outbound_delay — average outbound delay
    event_count         — number of turnaround events observed in this cell

The output table is the dynamic signal over time for every node in the airport graph,
ready to be sliced into lookback windows for the Temporal GNN.

Output: data/temporal_node_states.csv
"""

import os
import pandas as pd
import numpy as np

CHAINS_PATH = os.path.join(os.path.dirname(__file__), "rotation_chains.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "temporal_node_states.csv")


def extract_hour(dep_time_blk: pd.Series) -> pd.Series:
    """Parse the leading hour from a DepTimeBlk string like '0600-0659'."""
    cleaned = dep_time_blk.astype(str).str.strip()
    hour = pd.to_numeric(cleaned.str[:2], errors="coerce").fillna(-1).astype(int)
    return hour


if __name__ == "__main__":
    print("Loading rotation chains...")
    chains = pd.read_csv(CHAINS_PATH, low_memory=False)
    print(f"  {len(chains):,} rotation events")

    chains["hour"] = extract_hour(chains["dep_time_blk"])

    # Per (airport, date, time-block) aggregation
    states = (
        chains.groupby(
            ["turnaround_airport", "FlightDate", "dep_time_blk", "DayOfWeek", "hour"],
            sort=False,
        )
        .agg(
            mean_alpha          = ("alpha",          "mean"),
            median_alpha        = ("alpha",          "median"),
            p75_alpha           = ("alpha",          lambda x: x.quantile(0.75)),
            mean_inbound_delay  = ("inbound_delay",  "mean"),
            mean_outbound_delay = ("outbound_delay", "mean"),
            event_count         = ("alpha",          "count"),
        )
        .reset_index()
    )

    states["FlightDate"] = pd.to_datetime(states["FlightDate"])
    states = states.sort_values(["turnaround_airport", "FlightDate", "hour"]).reset_index(drop=True)

    print(f"\nTemporal states shape : {states.shape}")
    print(f"Date range            : {states['FlightDate'].min().date()} to "
          f"{states['FlightDate'].max().date()}")
    print(f"Airports              : {states['turnaround_airport'].nunique()}")
    print(f"Time blocks           : {states['dep_time_blk'].nunique()}")
    print(f"\nmean_alpha summary")
    print(states["mean_alpha"].describe().to_string())

    states.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
