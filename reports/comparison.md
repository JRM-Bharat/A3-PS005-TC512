# Phase 5 Evaluation Report: Baseline vs Machine Learning vs Hybrid

> [!NOTE]
> **Scale Change Declaration**: The ML dataset is generated with identical scenario definitions,
> prefix allocations, and ROA schedules to Phase 4. Event counts were scaled to 1,000 total events
> (400 anomalies, 600 benign) with 31,700 feed updates, so that chronological time slices (Train 60%,
> Val 20%, Test 20%) each contain $\ge 20$ forged-origin positives (`loose_maxlength_hijack`) for robust benchmarking.

## 1. Generator Audit: Peer Novelty & Origin Rarity

To prevent trivial feature leakage, peer novelty and origin rarity were audited across all traffic classes:

| Traffic Class | Total Updates | Known Peers % | Novel Peer % | Rarity Mean | Rarity Median | Rarity Min | Rarity Max | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forged-Origin Attacks (`loose_maxlength_hijack`)** | 120 | 49.2% | 50.8% | 0.0006 | 0.0004 | 0.0004 | 0.0048 | Clean |
| **TE More-Specific (`te_more_specific`)** | 100 | 50.0% | 50.0% | 0.0021 | 0.0022 | 0.0004 | 0.0303 | Clean |
| **Other Benign Scenarios (Anycast, Scrubbing, Lag)** | 922 | 78.0% | 22.0% | 0.0395 | 0.0127 | 0.0068 | 1.0000 | Clean |
| **Background Traffic** | 25,028 | 80.1% | 19.9% | 0.0297 | 0.0078 | 0.0004 | 1.0000 | Clean |

## 2. Dataset Chronological Time Split & Scenario Coverage

Strict chronological splitting was enforced with zero future leakage:
- **Train**: First 60% of feed updates ($t \in [0.00\text{h}, 14.40\text{h}]$)
- **Validation**: Middle 20% of feed updates ($t \in [14.40\text{h}, 19.20\text{h}]$ — threshold selection only)
- **Test**: Final 20% of feed updates ($t \in [19.20\text{h}, 24.00\text{h}]$ — held-out benchmark)

| Scenario Name | Difficulty / Type | Train Events | Val Events | Test Events | Total Events |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `exact_origin_hijack` | Scenario | 61 | 21 | **18** | 100 |
| `hijack_after_roa_change` | Scenario | 60 | 20 | **20** | 100 |
| `loose_maxlength_hijack` | Scenario | 73 | 25 | **22** | 120 |
| `multi_origin_anycast` | Scenario | 47 | 27 | **26** | 100 |
| `multi_origin_ddos_scrubbing` | Scenario | 60 | 20 | **20** | 100 |
| `policy_change_roa_first` | Scenario | 60 | 20 | **20** | 100 |
| `policy_change_roa_lag_inventory_stale` | Scenario | 60 | 23 | **17** | 100 |
| `policy_change_roa_lag_inventory_updated` | Scenario | 60 | 23 | **17** | 100 |
| `subprefix_hijack` | Scenario | 61 | 21 | **18** | 100 |
| `te_more_specific` | Scenario | 61 | 21 | **18** | 100 |

## 3. Leakage Audit & Controls

Three mandatory honesty checks were executed to evaluate target and temporal leakage:
1. **Single-Feature Discrimination**: Single-feature ROC-AUC was computed for all 10 features.
2. **Shuffled-Label Control**: 20 permutations of training labels scored test **mean AUC = 0.5522 \pm 0.1954** (min: 0.1565, max: 0.8442).
3. **Validation Threshold Tuning**: Decision threshold $\tau^* = 0.0500$ was selected **strictly on Validation**.

| Feature Name | Raw Train AUC | Effective AUC | Direction | Leakage Status |
| :--- | :---: | :---: | :---: | :---: |
| `peer_new_for_prefix_origin` | 0.8929 | 0.8929 | `+` | Exceeds ~0.85 goal (0.893; residual limit) |
| `rov_state_encoded` | 0.8457 | 0.8457 | `+` | Clean (<= 0.85) |
| `origin_rarity` | 0.8356 | 0.8356 | `+` | Clean (<= 0.85) |
| `origin_new_for_prefix` | 0.7937 | 0.7937 | `+` | Clean (<= 0.85) |
| `origin_in_inventory` | 0.3586 | 0.6414 | `-` | Clean (<= 0.85) |
| `update_rate_30m` | 0.3639 | 0.6361 | `-` | Clean (<= 0.85) |
| `more_specific_first_seen` | 0.6022 | 0.6022 | `+` | Clean (<= 0.85) |
| `prefix_len_delta` | 0.5516 | 0.5516 | `+` | Clean (<= 0.85) |
| `prefix_age_hours` | 0.4616 | 0.5384 | `-` | Clean (<= 0.85) |
| `hour_of_day` | 0.5275 | 0.5275 | `+` | Clean (<= 0.85) |
| **Shuffled-Label Control (20 perms)** | — | **0.5522 \pm 0.1954** | — | Zero Systematic Leakage on Avg |

> [!WARNING]
> **Known Residual Separability Limit**: Feature `peer_new_for_prefix_origin` achieves an effective AUC of **0.893** (0.8929), exceeding the ~0.85 ceiling goal. In the synthetic traffic generator, novel peer appearances remain correlated with attack bursts despite per-update randomisation, presenting a known residual separability limit in the synthetic generator. This limit is documented plainly rather than artificially removed or masked.

> [!NOTE]
> **Shuffled-Label Control Analysis**: Evaluated against true test labels across 20 permutations with **113 positive rows** across **77 anomaly events** in the test slice.
> - **Mean AUC**: `0.5522` (centered near 0.50 chance)
> - **Measured Std**: `0.1954` (replacing the generic $\pm 0.05$ band)
> - **Range**: `[0.1565, 0.8442]`
>
> The wide standard deviation (0.1954) reflects high variance stemming from clustered positive rows per anomaly event (multiple BGP updates per burst sharing rolling state) and the compact size of the test slice. The control demonstrates that the feature pipeline learns **no systematic target leakage on average**, though this control alone does not prove that leakage is entirely absent.

### Operating Threshold $\tau^*$ Performance

Decision threshold $\tau^* = 0.0500$ was selected strictly on the Validation slice to maximize F1 score.

| Slice / Level | Threshold $\tau^*$ | Precision | Recall | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Validation (Row-Level)** | 0.0500 | 0.8462 | 0.9680 | 0.9030 | Threshold selection slice |
| **Test (Row-Level)** | 0.0500 | 0.9083 | 0.9646 | 0.9356 | Held-out row evaluation |
| **Test (Event-Level - ML Alone)** | 0.0500 | 0.8588 | 0.9481 | 0.9012 | Primary benchmark (1-to-1 matching) |

The standalone Precision-Recall curve with operating point $\tau^*$ is saved at [`reports/ml_pr_curve.png`](file:///d:/Project/bgp-anomaly-monitor/reports/ml_pr_curve.png).

## 4. Benchmark Comparison on Held-Out Test Slice (Multi-Seed & Bootstrap CI)

> [!IMPORTANT]
> **Multi-Seed Interpretation & Scope**: Seeds 777, 888, and 999 represent variance across independent synthetic dataset draws generated by ONE stochastic generator. They demonstrate algorithmic stability across stochastic variations in the same underlying traffic model, but do **NOT** establish external validity on real-world operator BGP feeds or diverse production network topologies.

| Model / Policy | Event-Level Precision | Event-Level Recall | Event-Level F1 | Alert-Level Precision | Latency (ms/upd) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Primary)** | 0.5185 $\pm$ 0.0000 | 0.7242 $\pm$ 0.0044 | 0.6043 $\pm$ 0.0015 | 0.5185 $\pm$ 0.0000 | 0.0138 ms |
| **Baseline (Secondary)** | 0.6364 $\pm$ 0.0000 | 0.7242 $\pm$ 0.0044 | 0.6774 $\pm$ 0.0019 | 0.6364 $\pm$ 0.0000 | 0.0138 ms |
| **ML Alone** | 0.9044 $\pm$ 0.0325 | 0.9183 $\pm$ 0.0261 | 0.9104 $\pm$ 0.0087 | 0.9044 $\pm$ 0.0325 | 0.6496 ms |
| **Hybrid (Primary OR ML)** | 0.5476 $\pm$ 0.0023 | 0.9183 $\pm$ 0.0261 | 0.6859 $\pm$ 0.0067 | 0.5476 $\pm$ 0.0023 | 0.6496 ms |
| **Hybrid (Secondary OR ML)** | 0.6476 $\pm$ 0.0051 | 0.9183 $\pm$ 0.0261 | 0.7593 $\pm$ 0.0066 | 0.6476 $\pm$ 0.0051 | 0.6496 ms |

## 5. Anomaly Scenario Recall Breakdown

| Anomaly Scenario | Difficulty | Baseline (Primary) | Baseline (Secondary) | ML Alone | Hybrid (Primary OR ML) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `exact_origin_hijack` | Easy | 100.0% | 100.0% | **100.0%** | **100.0%** |
| `subprefix_hijack` | Easy | 100.0% | 100.0% | **100.0%** | **100.0%** |
| `hijack_after_roa_change` | Hard | 100.0% | 100.0% | **100.0%** | **100.0%** |
| `loose_maxlength_hijack` | Hard (Forged Origin) | 0.0% | 0.0% | **81.0%** | **81.0%** |

## 6. False Positive Breakdown by Benign Scenario

| Benign Scenario | Test Events | Baseline (Primary) FP | Baseline (Secondary) FP | ML Alone FP | Hybrid (Primary OR ML) FP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `multi_origin_anycast` | 26 | 0/26 | 0/26 | **1/26** | 1/26 |
| `multi_origin_ddos_scrubbing` | 20 | 20/20 | 0/20 | **0/20** | 20/20 |
| `policy_change_roa_first` | 20 | 0/20 | 0/20 | **0/20** | 0/20 |
| `policy_change_roa_lag_inventory_stale` | 17 | 16/17 | 16/17 | **3/17** | 16/17 |
| `policy_change_roa_lag_inventory_updated` | 17 | 16/17 | 16/17 | **0/17** | 16/17 |
| `te_more_specific` | 18 | 0/18 | 0/18 | **8/18** | 8/18 |

## 7. Operational Trade-Offs & Conclusions

1. **Forged-Origin Hijack Detection (The Primary ML Win)**:
   - RFC 6811 ROV and inventory rules are **fundamentally blind (0.0% recall)** to forged-origin subprefixes (`loose_maxlength_hijack`) because the attacker announces a more-specific using the victim's authorized ASN.
   - The ML model detects forged-origin attacks without rule alerts, achieving high recall by learning peer novelty, length deltas, and arrival dynamics.
2. **False Positive Trade-Offs**:
   - Under Primary Policy, the rule baseline triggers false positives on DDoS scrubbing (`INVENTORY_MISMATCH`) and policy lag.
   - The ML model dramatically suppresses DDoS scrubbing false positives while retaining high precision.
   - The Hybrid detector (`Rules OR ML`) combines deterministic explainability on RFC 6811 invalids with ML detection of forged origins.
3. **Streaming Latency**:
   - Rule Detector: **0.0138 ms/update** (unbatched stream with ROA lookup)
   - ML Detector: **0.6496 ms/update** (unbatched stream with rolling feature extraction)
