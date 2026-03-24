"""
Build static airport graph: nodes (FAA features + derived hub tier)
and directed edges (T-100 route frequency / passenger volume).

Outputs:
    data/graph_nodes.csv  — one row per airport
    data/graph_edges.csv  — one row per directed route
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(__file__)
T100_PATH  = os.path.join(DATA_DIR, "datasets", "100T", "T_T100_SEGMENT_ALL_CARRIER.csv")
FAA_PATH   = os.path.join(DATA_DIR, "datasets", "FAA", "Airports.csv")
ONTIME_PATH = os.path.join(DATA_DIR, "merged_ontime_2025.csv")

NODES_OUT = os.path.join(DATA_DIR, "graph_nodes.csv")
EDGES_OUT  = os.path.join(DATA_DIR, "graph_edges.csv")

# Hub tier thresholds (by rank of total scheduled departures)
MAJOR_TOP_N    = 30
REGIONAL_TOP_N = 100


def build_edges(ontime_airports: set) -> pd.DataFrame:
    print("Loading T-100 segment data...")
    t100 = pd.read_csv(T100_PATH, low_memory=False)
    print(f"  Raw rows: {len(t100):,}")

    # Keep domestic scheduled passenger service only
    t100 = t100[
        (t100["CLASS"] == "F") &
        (t100["ORIGIN_COUNTRY"] == "US") &
        (t100["DEST_COUNTRY"] == "US") &
        t100["ORIGIN"].isin(ontime_airports) &
        t100["DEST"].isin(ontime_airports)
    ].copy()
    print(f"  After domestic/scheduled filter: {len(t100):,} rows")

    edges = (
        t100.groupby(["ORIGIN", "DEST"])
        .agg(
            route_frequency   = ("DEPARTURES_PERFORMED", "sum"),
            passenger_volume  = ("PASSENGERS",           "sum"),
            distance_miles    = ("DISTANCE",             "mean"),
        )
        .reset_index()
        .rename(columns={"ORIGIN": "origin", "DEST": "dest"})
    )

    edges = edges[edges["route_frequency"] > 0].copy()
    edges["distance_miles"] = edges["distance_miles"].round(1)
    print(f"  Directed edges: {len(edges):,}")
    return edges


def build_nodes(ontime_airports: set, edges: pd.DataFrame) -> pd.DataFrame:
    print("\nLoading FAA airport data...")
    faa = pd.read_csv(FAA_PATH, low_memory=False)

    # Operational airports with 3-letter codes (IATA)
    faa = faa[
        (faa["OPERSTATUS"] == "OPERATIONAL") &
        (faa["IDENT"].str.len() == 3)
    ][["IDENT", "NAME", "X", "Y", "ELEVATION", "STATE", "TYPE_CODE"]].copy()
    faa.columns = ["airport", "name", "longitude", "latitude", "elevation_ft", "state", "type_code"]
    faa = faa.drop_duplicates("airport")

    nodes = pd.DataFrame({"airport": sorted(ontime_airports)})
    nodes = nodes.merge(faa, on="airport", how="left")

    # Backfill state for the ~10 airports missing from FAA via T-100 metadata
    t100_state = (
        pd.read_csv(T100_PATH, usecols=["ORIGIN", "ORIGIN_STATE_ABR"], low_memory=False)
        .drop_duplicates("ORIGIN")
        .rename(columns={"ORIGIN": "airport", "ORIGIN_STATE_ABR": "state_t100"})
    )
    nodes = nodes.merge(t100_state, on="airport", how="left")
    missing_state = nodes["state"].isna()
    nodes.loc[missing_state, "state"] = nodes.loc[missing_state, "state_t100"]
    nodes.drop(columns=["state_t100"], inplace=True)

    # Hub tier derived from total T-100 scheduled departures
    total_deps = (
        edges.groupby("origin")["route_frequency"]
        .sum()
        .reset_index()
        .rename(columns={"origin": "airport", "route_frequency": "total_departures"})
    )
    nodes = nodes.merge(total_deps, on="airport", how="left")
    nodes["total_departures"] = nodes["total_departures"].fillna(0).astype(int)

    rank = nodes["total_departures"].rank(ascending=False, method="min")
    nodes["hub_tier"] = "local"
    nodes.loc[rank <= REGIONAL_TOP_N, "hub_tier"] = "regional"
    nodes.loc[rank <= MAJOR_TOP_N,    "hub_tier"] = "major"

    # Node index for GNN adjacency matrix construction later
    nodes = nodes.reset_index(drop=True)
    nodes.index.name = "node_idx"
    nodes = nodes.reset_index()

    print(f"  Nodes: {len(nodes):,}")
    print(f"  Hub tiers: {nodes['hub_tier'].value_counts().to_dict()}")

    faa_matched = nodes["latitude"].notna().sum()
    print(f"  FAA-matched: {faa_matched} / {len(nodes)} airports "
          f"({faa_matched / len(nodes):.1%})")
    return nodes


if __name__ == "__main__":
    print("Reading On-Time airport universe...")
    ot = pd.read_csv(ONTIME_PATH, usecols=["Origin", "Dest"], low_memory=False)
    ontime_airports = set(ot["Origin"].unique()) | set(ot["Dest"].unique())
    print(f"  {len(ontime_airports)} unique airports\n")

    edges = build_edges(ontime_airports)
    nodes = build_nodes(ontime_airports, edges)

    edges.to_csv(EDGES_OUT, index=False)
    nodes.to_csv(NODES_OUT, index=False)
    print(f"\nSaved: {EDGES_OUT}")
    print(f"Saved: {NODES_OUT}")
