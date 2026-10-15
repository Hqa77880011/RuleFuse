# RuleFuse

Implementation of **RuleFuse: Conflict-Provenance Reasoning for Multi-Sensor Measurement Fusion under Missing and Contradictory Evidence**.

RuleFuse retains missing-channel ignorance, computes inclusion-minimal inconsistent evidence subsets, compares sensor-origin and physical multi-fault explanations, and fuses measurements after provenance routing. Every case records evidence, conflicts, hypothesis scores, residual conflict, diagnosis, and a minimum-expected-cost action.

## Install and prepare

```bash
python -m venv .venv
python -m pip install -e .
```

Python 3.11+ is required. Obtain CWRU or Paderborn recordings from their original bearing-data providers. Extract the same per-channel symptom features for all compared fusion methods: examples include RMS, kurtosis, and declared defect-frequency band energies. Keep source recordings/bearings in one partition, preserving cross-load or cross-speed holdouts. Do not treat artificially superposed faults as naturally co-occurring field failures.

The input NPZ contains:

| Array | Shape and meaning |
| --- | --- |
| `x` | `N,S,D` finite channel feature vectors; unobserved entries may be zero because the separate mask controls their use |
| `mask` | `N,S` boolean availability |
| `y` | `N` indices into ontology physical worlds |
| `split` | `N` strings `train`, `val`, `test` |
| `group` | `N` original recording or bearing identifiers |
| `origin` | Optional `N` strings `physical` or `sensor`, for provenance scoring |
| `bad_sensor` | Optional `N` corrupted sensor index, or -1 when no sensor failed |

```bash
rulefuse prepare --source features.npz --output cases.npz
rulefuse fit --data cases.npz --ontology ontology.json --output evidence_model.json
```

Instead of `x`, the source NPZ may provide `signal` with shape `N,S,L` containing segmented raw vibration windows. Preparation computes RMS, kurtosis, crest factor, and three equal-FFT-band log energies, then checks partitions and group exclusivity. For defect-frequency-specific engineering features, provide your registered `x` directly. Fitting trains one Gaussian symptom encoder per sensor, fits its temperature on validation observations, and estimates reliability from validation correctness. Every represented physical world needs training observations in every sensor. Use exactly the same fitted evidence model for matched fusion comparisons.

`ontology.json` contains a finite physical-state table: each world is a fault-presence bit vector. The supplied two-fault ontology includes healthy, two single faults, and their compound. Replace it with the actual taxonomy and engineering maintenance costs. Each `action_cost[a][f]` specifies action `a` under physical world `f`, with optional fixed `inspection_cost[a]`. The supplied cost units are an explicit example convention.

## Inference, baselines, and ablations

```bash
rulefuse evaluate --data cases.npz --model evidence_model.json --output outputs/full
rulefuse evaluate --data cases.npz --model evidence_model.json --output outputs/late --variant late_fusion
rulefuse evaluate --data cases.npz --model evidence_model.json --output outputs/no_provenance --variant no_provenance
rulefuse evaluate --data cases.npz --model evidence_model.json --output outputs/no_rules --variant no_rules
rulefuse evaluate --data cases.npz --model evidence_model.json --output outputs/impute --variant impute
```

Available sensors assert their most supported symptom. A nonhealthy symptom is compatible with physical worlds containing its active faults; a healthy assertion is compatible with healthy only. The initial single-fault rule frame is used to identify every inclusion-minimal inconsistent subset. Hypotheses either admit compound physical worlds or discount one implicated source to 5% of its original reliability. They are scored by unexplained weighted conflict, new contradictions, and evidence likelihood, then marginalized into physical-state probabilities. Missing sensors assign all belief to ignorance and contribute no assertion. The `--penalty` coefficient defaults to 2; choose it on validation cases.

`late_fusion` is reliability-weighted probability averaging, `no_provenance` retains conflict/rule penalties but disables hypothesis routing, `no_rules` removes consistency penalties, and `impute` fills unavailable evidence from available-channel mean probabilities before fusion. None of these baselines is labeled as a different published algorithm.

`cases.json` exposes the full decision object, including all minimal conflicts and each provenance hypothesis. `metrics.json` includes physical-state Macro-F1, compound-case multilabel micro-F1, ECE, multiclass Brier score, selected-world rule violations, realized maintenance cost, and explanation fidelity/size. Conflict-source accuracy and sensor localization are computed only when the corresponding true annotations are present. Explanation fidelity is the selected hypothesis's resolved fraction of weighted conflict; size counts assertion incidences across minimal conflicts. `confidence.png` plots correctness against fused confidence.

To measure missingness robustness, construct paired case files by changing `mask` while keeping features and labels fixed. Sensor-bias experiments alter only declared observed sensor features; compound experiments use a registered superposition protocol and distinct physical labels. Wrong-rule experiments change the ontology, keeping cases fixed. The generic manuscript does not supply an executable rule ontology or provenance likelihood. This implementation makes both explicit through finite fault states, symptom compatibility, and a Gibbs posterior over the displayed conflict score. Minimal-conflict enumeration is exact and exponential in sensor count, so use a small diagnostic sensor set. No experiment outcomes are included.
