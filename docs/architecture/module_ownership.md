# Module Ownership Map

| Area | Path | Owner role | Contract |
|------|------|------------|----------|
| Runtime orchestrator | `ml/runtime/` | Platform | OpenAPI + `ml/runtime/contracts.py` |
| Vision model | `ml/models/` | ML | Tier configs + export parity |
| Training | `ml/training/` | ML Ops | Observability + reproducibility gates |
| Therapy | `ml/therapy/` | Therapy/ML | Scoring + safety constraints |
| RAG | `ml/retrieval/`, `ml/pipeline/` | Retrieval | RAG SLO + guard layer |
| Data | `ml/data/` | Data Eng | Ontology + ingest validators |
| Infra | `ml/infra/` | Infra | IAM/KMS/SageMaker policies |
| App UX | `app/` | Client | Haptics, personal mode, device bridge |
| Simulator | `tools/simulation/` | Platform | Dev-only; must not diverge from runtime API |
| Product CLI | `scripts/product/` | Platform | Single canonical entry point |

Every PR touching a contract module must update tests in `tests/test_runtime_contracts.py` or the phase-specific suite.
