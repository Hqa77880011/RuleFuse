# RuleFuse

Implementation of **RuleFuse: Conflict-Provenance Reasoning for Multi-Sensor Measurement Fusion under Missing and Contradictory Evidence**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

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

Instead of `x`, the source NPZ may provide `signal` with shape `N,S,L` containing segmented raw vibration windows. Preparation computes RMS, kurtosis, crest factor, and three equal-FFT-band log energies, then checks partitions and group exclusivity. For defect-frequency-specific engineering features, provide your registered `x` directly. Fitting trains one Gaussian symptom encoder per sensor, fits its temperature on validation observations, and estimates reliability from validation correctness. Every represented physical world needs training observations in every sensor. Use exactly the same fitted evidence model for matched fusion comparisons.

`ontology.json` contains a finite physical-state table: each world is a fault-presence bit vector. The supplied two-fault ontology includes healthy, two single faults, and their compound. Replace it with the actual taxonomy and engineering maintenance costs. Each `action_cost[a][f]` specifies action `a` under physical world `f`, with optional fixed `inspection_cost[a]`. The supplied cost units are an explicit example convention.

```bash
rulefuse prepare --source features.npz --output cases.npz
```

## Training and inference

```bash
rulefuse fit --data cases.npz --ontology ontology.json --output evidence_model.json
```

## Evaluation

```bash
rulefuse evaluate --data cases.npz --model evidence_model.json --output outputs/full
```

## Ablations

Supported controls: `late_fusion`, `no_provenance`, `no_rules`, `impute`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
rulefuse evaluate --data cases.npz --model evidence_model.json --output outputs/late_fusion --variant late_fusion
```

## Outputs

`evidence_model.json` stores fitted sensor evidence. Evaluation writes `cases.json`, `metrics.json` and `confidence.png`.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
