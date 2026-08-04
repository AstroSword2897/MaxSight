# Stage A embedded verification keys

## Purpose

On-device (Python reference) verification trusts only keys shipped in the app/binary package. Keys are never fetched over the network.

## Trust window

- `current.pub` / `current.hmac`: active verification material
- `next.pub` / `next.hmac`: rotation candidate accepted during overlap

Verification succeeds if **either** current or next verifies the artifact signature.

## Rotation runbook

1. Generate next key material offline in CI.
2. Ship `next.*` in an app update while `current.*` remains valid.
3. After overlap, promote next → current in a subsequent app release.
4. Compromised-key revocation requires an app update (app-store latency is an accepted bound).
