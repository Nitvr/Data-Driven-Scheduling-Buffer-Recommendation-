Project Proposal: Data-Driven Scheduling Buffer Recommendation Using Temporal GNNs on Airport Networks
Motivation
Flight delays are very costly to both airlines and the US economy as a whole. Connected flights and airports often mean that individual delays can often propagate and lead to further delays downstream. A major reason of this systemic delay is the insufficient turnaround buffers at airports, where later-than-expected flights amplify existing delays and into outbounding flights
Traditional scheduling approaches use fixed minimum turnaround times that do not account for real-time network conditions or airport-specific delay dynamics. Recent advances in temporal Graph Neural Networks (GNNs) have demonstrated strong performance in dynamic network prediction tasks such as traffic forecasting. Despite this success, GNNs have not yet been systematically applied to recommend adaptive scheduling buffers in airline networks.
This project applies temporal GNN architectures to the problem of scheduling buffer recommendation, treating the airport network as a dynamic graph where delay signals — and their amplification — propagate across edges (routes) over time. By learning delay amplification factors for each airports and departure time, the model outputs a policy table of recommended minimum turnaround buffers that minimizes propagated delay across the network.
Problem Statement
Given:
•	A directed graph G=(V,E) representing the US domestic airport network, where each node V represents an airport and each edge represents an airline route between airports.
•	Historical flight delay and turnaround time series data for each airport and route, including inbound delay, scheduled turnaround duration, and outbound departure delay.
•	Static node features (airport capacity, hub tier, timezone) and edge features (route frequency, passenger volume).
•	Time context features such as hour-of-the-day and day-of-the-week for each observation.
For each airport, we record the amount of delay introduced by the aircraft as it arrives as the inbounrd delay; we also record the amount of delay experienced by the aircraft before its next departure at the airport as outbound delay. We then define a delay amplification factor α as the ratio of outbound delay introduced relative to inbound delay absorbed for each turnaround event. The goal is to recommend the minimum buffer that reduces α below a target threshold.
Predict:
•	The delay amplification factor αᵢ⁼ᵗ at each airport node i and time-of-day window t, given current and historical delay states across the graph.
•	The recommended minimum turnaround buffer Bᵢ⁼ᵗ (in minutes) such that the expected amplification αᵢ⁼ᵗ falls below a specified threshold, output as a policy table by airport × time-of-day.
Formally, at time t, each node vi has a delay state di(t) representing the average inbound delay in the past hour. The GNN learns a function f:
f : (G, {d⁼ᵏ,...,dᵗ}) → {α⁼ᵗ⁺¹,...,α⁼ᵗ⁺ʰ} → {B⁼ᵗ⁺¹,...,B⁼ᵗ⁺ʰ}
Datasets
•	BTS On-Time Performance Dataset: Provides per-flight scheduled and actual departure/arrival times, enabling computation of inbound delay, turnaround duration, and outbound delay for every aircraft rotation.
•	BTS T-100 Segment Data: Supplies monthly route-level traffic statistics used to construct graph edges with frequency and volume features.
•	FAA Airport Data: Provides static node features including airport capacity, hub classification, and timezone.
Approach
The project proceeds in three modeling stages:
Stage 1 — Amplification Factor Learning
Using the BTS On-Time data, we construct aircraft rotation chains to pair each inbound flight with its subsequent outbound departure from the same aircraft. For every turnaround event we compute α = max(0, outbound_delay − inbound_delay) / max(1, inbound_delay).
Stage 2 — Temporal GNN Model
Train a temporal GNN on the dynamic airport graph to predict α at each node for the next time window. The model takes in the airport network graph structure and a historical lookback window of delay states to learn how delay amplification at one airport is influenced by the delays of neighboring airports.
Stage 3 — Buffer Policy Generation and Simulation
Using the trained model, we sweep candidate buffer values for each airport × time-of-day cell and predict the resulting α. The minimum buffer that reduces α below a target threshold (e.g., α < 0.2) is recorded in a policy table. 
Success Metrics
•	MAE and RMSE on predicted amplification factor α at each airport node.
•	Classification accuracy for high-amplification events (α > 0.5 and α > 1.0), indicating turnarounds that significantly worsen delays.
•	Policy simulation benefit: total propagated delay minutes reduced under the GNN-recommended buffer policy vs. historical fixed buffers.
Timeline
•	Weeks 1–2: Data collection and preprocessing (download BTS On-Time and T-100 data, construct aircraft rotation chains, compute α labels, build airport graph).
•	Weeks 3–4: Exploratory analysis and feature engineering (analyze amplification distributions by airport and time-of-day, build node/edge features).
•	Weeks 5–6: Baseline model implementation (per-airport regression and global buffer benchmarks).
•	Weeks 7–8: Temporal GNN model implementation and training (implement TGCN/STGCN, tune hyperparameters, train on amplification prediction task).
•	Week 9-10: Policy table generation, and simulation of delay savings. Finalize report
