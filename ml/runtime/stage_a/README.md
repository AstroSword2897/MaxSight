# Stage A hard contract (ADR)

## Status

Accepted for MaxSight Phone MVP (Python reference implementation).

## Context

Stage A (hazard / urgency / distance) must run correctly even if the phone has never made a network call. Prior drafts treated this as developer discipline; that is insufficient for a safety-critical path.

## Decision

`ml/runtime/stage_a/` is a formal module boundary:

- `StageARunner.infer(frame) -> HazardResult` is the only inference entrypoint.
- The interface accepts no network client, connectivity flag, or cloud callback.
- CI enforces isolation via `scripts/infra/validate_stage_a_isolation.py`.
- Model activation uses bundle-local `ACTIVE_MODEL_PTR` / `LAST_KNOWN_GOOD` resolution with fail-closed refuse messaging — never silent degradation.

App-layer concerns (connectivity, Stage B, OTA) live under `app/` and must not be importable into this package.

## Consequences

Native Android/iOS runners will bind to this contract later. Python `TorchStageARunner` is the canonical reference for tests. Any change that would reshape `HazardResult` / `CameraFrame` / `StageARunner` requires a deliberate contract PR — ML wiring adapts to the contract, not the reverse.
