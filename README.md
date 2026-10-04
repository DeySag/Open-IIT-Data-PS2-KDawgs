# PS2 RPC: Right-Party Contact Prediction and Skip-Trace Prioritisation

CreditNirvana collections platform component for predicting contact-point health and prioritising skip-traces.

## Structure

```
.
├── configs/              # Configuration files (sim, guardrails, costs, field_mappings)
├── data/                 # Gitignored - generated datasets
├── docs/                 # assumptions.md, decision_log.md, design.md
├── src/rpc/
│   ├── contracts/        # Pydantic schemas: input envelope, output decision, suppression
│   ├── sim/              # Simulator + mock CN API / replay writer
│   ├── ingest/           # Adapters, field mappings, dedupe, event store
│   ├── features/         # Point-in-time pipeline, text extraction, graph features
│   ├── models/           # State tracker, reach latent, slot GBM, recycled, third-party, baselines
│   ├── decision/         # Guardrails, actions, reason codes, VOI, exploration
│   ├── serve/            # FastAPI app
│   └── eval/             # Splits, metrics, OPE, reports
└── tests/
```

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Generate dev dataset (~5k borrowers)
make data-dev

# Run baselines
make train-baselines

# Run end-to-end smoke test
make smoke
```

## Configuration

All tunable parameters live in `configs/`:
- `sim.yaml` - Simulator assumptions
- `guardrails.yaml` - Compliance rules
- `costs.yaml` - Channel and trace costs
- `field_mappings/` - Per-source adapters

## Contracts (Frozen)

- Input event envelope: `src/rpc/contracts/input.py`
- Output decision: `src/rpc/contracts/output.py`
- Suppression entry: `src/rpc/contracts/suppression.py`

## Documentation

- `docs/assumptions.md` - Working assumptions awaiting CN confirmation
- `docs/decision_log.md` - Architectural decisions with rationale
- `docs/design.md` - Technical design document