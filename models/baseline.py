"""
Weeks 5-6: Baseline α prediction models and buffer policy table.

Baselines (in increasing sophistication):
  1. Global mean          — predict grand-mean α for every event
  2. Per-airport mean     — predict per-airport historical mean α
  3. Per-airport × time-of-day mean — finest-grained lookup table
  4. Ridge regression     — cyclically-encoded time features + target-encoded airport

Evaluation metrics (per project proposal):
  - MAE  / RMSE on predicted α
  - Binary accuracy for high-amplification events: α > 0.5 and α > 1.0

Temporal train/test split:
  - Train: January – October 2025  (months 1-10)
  - Test:  November – December 2025 (months 11-12)

Buffer policy table:
  For each (airport, dep_time_blk) cell in the training set we compute the
  minimum buffer B that would have reduced each event's α below 0.2:

      B_min = max(0, outbound_delay − 1.2 × inbound_delay)

  (derived from: α(B) < 0.2  ⟺  outbound_delay − B < 1.2 × inbound_delay)

  The recommended buffer for that cell is the p80 of B_min values,
  meaning 80 % of historical events would have been absorbed.

Outputs:
  models/baseline_metrics.csv
  models/baseline_policy_table.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAINS_PATH = os.path.join(ROOT, "data", "rotation_chains.csv")
NODES_PATH  = os.path.join(ROOT, "data", "graph_nodes.csv")
METRICS_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_metrics.csv")
POLICY_OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_policy_table.csv")

TRAIN_MONTHS = list(range(1, 11))
TEST_MONTHS  = [11, 12]
BUFFER_PCTILE  = 0.80   # design buffer to cover this fraction of B_min values
MIN_CELL_EVENTS = 10   # discard cells with fewer events (too few observations to be reliable)


# ─── helpers ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    acc_05 = np.mean((y_true > 0.5) == (y_pred > 0.5))
    acc_10 = np.mean((y_true > 1.0) == (y_pred > 1.0))
    row = {"model": name, "MAE": round(mae,4), "RMSE": round(rmse,4),
           "Acc_α>0.5": round(acc_05,4), "Acc_α>1.0": round(acc_10,4)}
    print(f"  {name:<50s}  MAE={mae:.4f}  RMSE={rmse:.4f}"
          f"  Acc(>0.5)={acc_05:.3f}  Acc(>1.0)={acc_10:.3f}")
    return row


def cyclical(values: np.ndarray, period: float):
    """Return sin and cos encoding for cyclically periodic values."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


# ─── data loading ───────────────────────────────────────────────────────────

def load_data():
    print("Loading rotation chains...")
    chains = pd.read_csv(CHAINS_PATH, low_memory=False)
    chains["FlightDate"] = pd.to_datetime(chains["FlightDate"])
    chains["month"] = chains["FlightDate"].dt.month
    chains["hour"]  = chains["dep_time_blk"].astype(str).str[:2]
    chains["hour"]  = pd.to_numeric(chains["hour"], errors="coerce").fillna(0).astype(int)
    print(f"  {len(chains):,} rotation events  |  "
          f"{chains['turnaround_airport'].nunique()} airports")

    nodes = pd.read_csv(NODES_PATH, low_memory=False)
    chains = chains.merge(
        nodes[["airport", "hub_tier", "total_departures"]],
        left_on="turnaround_airport", right_on="airport", how="left"
    )
    return chains


# ─── feature engineering ────────────────────────────────────────────────────

def make_features(df: pd.DataFrame, airport_enc: pd.Series,
                  hub_enc: dict) -> np.ndarray:
    """Build feature matrix for Ridge regression."""
    hour_sin, hour_cos = cyclical(df["hour"].values.astype(float), 24.0)
    dow_sin,  dow_cos  = cyclical(df["DayOfWeek"].fillna(4).values.astype(float), 7.0)
    airport_feat = df["turnaround_airport"].map(airport_enc).fillna(airport_enc.mean()).values
    hub_feat     = df["hub_tier"].map(hub_enc).fillna(0).values
    return np.column_stack([hour_sin, hour_cos, dow_sin, dow_cos,
                             airport_feat, hub_feat])


# ─── buffer policy ──────────────────────────────────────────────────────────

def compute_policy(train: pd.DataFrame) -> pd.DataFrame:
    """
    For each (airport, time-block) cell, compute the recommended minimum
    turnaround buffer B (in minutes) at the p80 coverage level.

    Cells with fewer than MIN_CELL_EVENTS observations are excluded; callers
    should fall back to the airport-level or global recommendation for those.
    """
    cell = (
        train
        .groupby(["turnaround_airport", "dep_time_blk"])
        .agg(
            recommended_buffer_min = ("B_min", lambda x: round(x.quantile(BUFFER_PCTILE), 1)),
            mean_alpha             = ("alpha",  "mean"),
            pct_alpha_below_02     = ("alpha",  lambda x: round((x < 0.2).mean(), 3)),
            event_count            = ("alpha",  "count"),
        )
        .reset_index()
        .rename(columns={"turnaround_airport": "airport"})
    )

    # Drop sparse cells and fall back to per-airport aggregate
    sparse_mask  = cell["event_count"] < MIN_CELL_EVENTS
    airport_agg  = (
        train[train["turnaround_airport"].isin(
            cell.loc[sparse_mask, "airport"].unique()
        )]
        .groupby("turnaround_airport")
        .agg(
            recommended_buffer_min = ("B_min", lambda x: round(x.quantile(BUFFER_PCTILE), 1)),
            mean_alpha             = ("alpha",  "mean"),
            pct_alpha_below_02     = ("alpha",  lambda x: round((x < 0.2).mean(), 3)),
            event_count            = ("alpha",  "count"),
        )
        .reset_index()
        .rename(columns={"turnaround_airport": "airport"})
    )
    # Assign each sparse cell's dep_time_blk the airport-level buffer
    sparse_cells = cell[sparse_mask].copy()
    sparse_cells = sparse_cells.drop(
        columns=["recommended_buffer_min", "mean_alpha", "pct_alpha_below_02"]
    ).merge(
        airport_agg[["airport", "recommended_buffer_min", "mean_alpha", "pct_alpha_below_02"]],
        on="airport", how="left"
    )

    policy = pd.concat(
        [cell[~sparse_mask], sparse_cells], ignore_index=True
    ).sort_values(["airport", "dep_time_blk"]).reset_index(drop=True)

    policy["data_source"] = "cell"
    policy.loc[
        policy["event_count"] < MIN_CELL_EVENTS, "data_source"
    ] = "airport_fallback"
    return policy


# ─── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chains = load_data()

    train = chains[chains["month"].isin(TRAIN_MONTHS)].copy()
    test  = chains[chains["month"].isin(TEST_MONTHS)].copy()
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}\n")

    y_test = test["alpha"].values
    results = []

    # ── 1. Global mean ───────────────────────────────────────────────────────
    global_mean = train["alpha"].mean()
    results.append(evaluate(y_test, np.full(len(test), global_mean), "Global mean"))

    # ── 2. Per-airport mean ──────────────────────────────────────────────────
    airport_mean = train.groupby("turnaround_airport")["alpha"].mean()
    pred2 = test["turnaround_airport"].map(airport_mean).fillna(global_mean).values
    results.append(evaluate(y_test, pred2, "Per-airport mean"))

    # ── 3. Per-airport × time-of-day mean ───────────────────────────────────
    cell_mean = train.groupby(["turnaround_airport", "dep_time_blk"])["alpha"].mean()

    def lookup_cell(row):
        key = (row["turnaround_airport"], row["dep_time_blk"])
        if key in cell_mean.index:
            return cell_mean[key]
        return airport_mean.get(row["turnaround_airport"], global_mean)

    pred3 = test.apply(lookup_cell, axis=1).values
    results.append(evaluate(y_test, pred3, "Per-airport × time-of-day mean"))

    # ── 4. Ridge regression ──────────────────────────────────────────────────
    hub_enc      = {"major": 2, "regional": 1, "local": 0}
    airport_enc  = train.groupby("turnaround_airport")["alpha"].mean()   # target encode

    X_train = make_features(train, airport_enc, hub_enc)
    X_test  = make_features(test,  airport_enc, hub_enc)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, train["alpha"].values)
    pred4 = np.maximum(0.0, ridge.predict(X_te_s))
    results.append(evaluate(y_test, pred4,
                             "Ridge regression (airport_enc + cyclical time + hub)"))

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(METRICS_OUT, index=False)
    print(f"\nMetrics saved to {METRICS_OUT}")

    # ── Buffer policy table ──────────────────────────────────────────────────
    policy = compute_policy(train)
    policy.to_csv(POLICY_OUT, index=False)

    print(f"Policy table saved to {POLICY_OUT}  ({len(policy):,} airport x time-block cells)")
    print(f"  Cell-level entries   : {(policy['data_source']=='cell').sum():,}")
    print(f"  Airport fallback     : {(policy['data_source']=='airport_fallback').sum():,}")
    print(f"\nBuffer policy summary (recommended_buffer_min):")
    print(policy["recommended_buffer_min"].describe().round(1).to_string())

    pct_zero = (policy["recommended_buffer_min"] == 0).mean()
    print(f"\nCells requiring no additional buffer : {pct_zero:.1%}")

    # Top cells by recommended buffer (cell-level only)
    print("\nTop 15 airport x time-block cells by recommended buffer (cell-level):")
    top = (
        policy[policy["data_source"] == "cell"]
        .nlargest(15, "recommended_buffer_min")
        [["airport", "dep_time_blk", "recommended_buffer_min", "mean_alpha", "event_count"]]
    )
    print(top.to_string(index=False))
