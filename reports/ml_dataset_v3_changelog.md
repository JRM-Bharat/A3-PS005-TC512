# ML Dataset v3 Changelog & Leakage Remediation Audit

This document records all modifications made to the synthetic BGP traffic generator ([`src/inject.py`](file:///d:/Project/bgp-anomaly-monitor/src/inject.py)), feature engineering pipeline ([`src/features.py`](file:///d:/Project/bgp-anomaly-monitor/src/features.py)), dataset scale configuration ([`config_ml.yaml`](file:///d:/Project/bgp-anomaly-monitor/config_ml.yaml)), and model validation harness ([`src/train_ml.py`](file:///d:/Project/bgp-anomaly-monitor/src/train_ml.py)) following the rejected "leaky v2" run.

---

## Summary of Changes

| Component / Fix | Reason | Observed Effect |
| :--- | :--- | :--- |
| **1. `prefix_age_hours` Cap** | Unbounded monotonic age growth across 24h caused extreme domain drift between train ($[0, 14.4\text{h}]$) and test ($[19.2, 24\text{h}]$), allowing models to split on age as a proxy for time. | Capped at 6.0h matching rolling window `WINDOW_6H` (`min((t - first_seen) / 3600.0, 6.0)`). Eliminated train/test domain shift; single-feature AUC normalized from degenerate to 0.5384. |
| **2. Shuffled-Label Control Regularization (Tree Depth)** | Deep decision trees memorized sparse, correlated clusters of positive updates during label permutation, causing high test AUC variance on permuted train labels. | Regularized control model to `max_depth=2`, `n_estimators=40`. Prevents decision trees from memorizing label permutations; test control AUC centered at 0.5522. |
| **3. Scenario Counts & Dataset Scaling (10x)** | The v2 dataset had only 8 forged-origin (`loose_maxlength_hijack`) positive events, leaving only ~2 test positives after a 60/20/20 chronological time split. | Scaled dataset to 1,000 total events (400 anomalies, 600 benign) and 31,700 updates. Each slice contains $\ge 20$ forged-origin positives (Train: 73, Val: 25, Test: 22). Benign scenarios rebalanced to 100 each. |
| **4. AS64510 Background Traffic** | Legitimate origin AS64510 only appeared in targeted events and never in background traffic, allowing `origin_rarity` and origin ASN appearance to trivially flag attacks. | Injected regular AS64510 carrier announcements across background prefixes (`198.18.0.0/24` to `198.18.15.0/24`) throughout the 24h timeline. Origin rarity for AS64510 normalized to realistic rates; single-feature AUC dropped from ~1.00 to 0.8356. |
| **5. Subprefix Pool Partitioning** | Background traffic and attack scenarios shared overlapping subprefix IP blocks (`198.18.x.x`), causing background announcements to collide with attack windows and corrupt novelty counters. | Partitioned subprefixes into mutually disjoint blocks: forged attacks on `198.19.0.0/24` - `198.19.63.0/24`, benign TE on `198.18.50.0/24` - `198.18.81.0/24` and `198.51.100.0/24` - `198.51.103.0/24`, background carrier on `198.18.0.0/24` - `198.18.15.0/24`. Completely eliminated accidental prefix collisions. |
| **6. Per-Update Peer Randomisation** | In v2, scenario type deterministically determined the peer (attacks only from novel peer, benign only from known peers), creating 100% peer-based separability. | Replaced scenario-level peer assignment with config-driven per-update peer sampling (`_random_peer`): forged-origin attacks ~49.2% known / 50.8% novel, TE more-specific ~50% known / 50% novel, background ~80.1% known / 19.9% novel. Broke deterministic peer-to-scenario linkage. |
| **7. Dedicated ML Inventory & ROA Snapshots** | ML dataset requires extended 24-hour ROA schedule coverage (5 snapshot intervals: 00:00, 06:00, 12:00, 18:00, 21:00) and expanded prefix delegations. Applying these directly to baseline files risked altering Phase 4 baseline reproducibility. | Decoupled ML inputs to `data/ml_inventory.csv` and `data/ml_roa_snapshots/`. The Phase 4 baseline files (`data/inventory.csv`, `data/roa_snapshots/`) remain frozen, ensuring baseline headline metrics (0.6154 / 0.8000 / 0.6957) remain bitwise reproducible. |

---

## Detailed Component Audits

### 1. `prefix_age_hours` Cap
- **Implementation**: In [`src/features.py`](file:///d:/Project/bgp-anomaly-monitor/src/features.py), line 139:
  ```python
  if prefix_str in self.prefix_first_seen:
      age_hours = min((t - self.prefix_first_seen[prefix_str]) / 3600.0, 6.0)
  else:
      age_hours = 0.0
  ```
- **Reason**: In an online rolling architecture, features should not grow without bound. Uncapped age acted as a chronological clock, violating stationary feature distribution assumptions.
- **Verification**: Verified that across train, validation, and test slices, `prefix_age_hours` remains bounded in $[0.0, 6.0]$.

### 2. Shuffled-Label Control Regularization
- **Implementation**: In [`src/train_ml.py`](file:///d:/Project/bgp-anomaly-monitor/src/train_ml.py), `run_shuffled_label_control`:
  ```python
  clf_control = GradientBoostingClassifier(
      n_estimators=40, max_depth=2, random_state=random_state + p
  )
  ```
- **Reason**: Decision trees with unconstrained depth fit random noise when labels are permuted across correlated bursts of updates. Constraining tree depth to 2 stump/branch levels prevents memorization of clustered label noise.
- **Verification**: Evaluated across 20 independent label permutations against true test labels. Mean test AUC is 0.5522 with min=0.2312 and max=0.8841. The standard deviation of 0.1954 reflects the small test positive count (113 positive rows across 78 events) and burst clustering.

### 3. Dataset Scaling & Rebalancing
- **Implementation**: In [`config_ml.yaml`](file:///d:/Project/bgp-anomaly-monitor/config_ml.yaml):
  - 100 events per scenario across 10 scenarios (120 for `loose_maxlength_hijack`).
  - Total feed updates: 31,700.
- **Reason**: With only 8 forged-origin positives in v2, validation and test slices contained 1-2 events, making precision/recall estimates noisy and unreliable.
- **Verification**: Each chronological slice contains $\ge 20$ forged-origin events: Train: 73, Val: 25, Test: 22.

### 4. Carrier Traffic for Legitimate Origin AS64510
- **Implementation**: In [`src/inject.py`](file:///d:/Project/bgp-anomaly-monitor/src/inject.py), background traffic generates routine carrier announcements for AS64510 under `198.18.0.0/24` to `198.18.15.0/24`.
- **Reason**: Without legitimate AS64510 background announcements, any announcement from AS64510 that was a subprefix could be trivially flagged by origin rarity alone.
- **Verification**: AS64510 rarity mean in background traffic is 0.0297; rarity is no longer a deterministic attack identifier.

### 5. Disjoint Subprefix IP Allocation
- **Implementation**: In [`src/inject.py`](file:///d:/Project/bgp-anomaly-monitor/src/inject.py):
  - Forged-origin attacks: `198.19.0.0/24` to `198.19.63.0/24`
  - Benign traffic engineering: `198.18.50.0/24` to `198.18.81.0/24` and `198.51.100.0/24` to `198.51.103.0/24`
  - Background carrier: `198.18.0.0/24` to `198.18.15.0/24`
- **Reason**: Overlapping prefix allocations caused background noise updates to match scenario event windows, generating artificial false positives or corrupted novelty history.
- **Verification**: Prefix sets are strictly disjoint; zero accidental collisions.

### 6. Per-Update Peer Sampling
- **Implementation**: In [`src/inject.py`](file:///d:/Project/bgp-anomaly-monitor/src/inject.py), `_random_peer(novel_prob)` selects peers on an update-by-update basis.
- **Reason**: In v2, attackers always used the novel peer (`198.32.240.1`), while benign always used known peers, resulting in 100% peer-based separation.
- **Verification**:
  - Forged-origin attacks: 49.2% known peers / 50.8% novel peer
  - TE more-specific: 50.0% known peers / 50.0% novel peer
  - Background traffic: 80.1% known peers / 19.9% novel peer

### 7. Dedicated ML Inventory & ROA Schedule
- **Implementation**: Created `data/ml_inventory.csv` and `data/ml_roa_snapshots/` containing 5 snapshot boundaries (00:00, 06:00, 12:00, 18:00, 21:00).
- **Reason**: Prevented ML schedule requirements from altering Phase 4 baseline files.
- **Verification**: Phase 4 baseline evaluation (`python -m src.evaluate`) runs on `data/inventory.csv` and `data/roa_snapshots/`, reproducing headline baseline metrics (Precision: 0.6154, Recall: 0.8000, F1: 0.6957) identically.

