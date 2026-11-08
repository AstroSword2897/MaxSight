# ADR-016: Signed artifacts; bundled last-known-good model; offline rollback; refuse when both models are invalid

- **Register ID:** DR-034
- **Status:** Confirmed
- **Source:** Q170–177

## Context / rationale
Integrity

## Alternatives considered
Unsigned

## Decision
Signed artifacts; bundled last-known-good model; offline rollback; refuse when both models are invalid

## Consequences
KMS + on-device verify

**Implementation note (2026-11-07):** on-device verification is an Ed25519 instance contract
(`docs/contracts/schemas/artifact_signature.json`) governed by `configs/security/artifact_signing.yaml`;
unsigned artifacts are refused. The earlier HMAC keys were committed to a public repository and are treated
as compromised. The build signs with an Ed25519 file key from the secret store; moving custody into KMS
(as this ADR anticipated) is an open decision.

## Dependencies
—

## Validation required
Integrity tests

## Approver
_pending signature_
