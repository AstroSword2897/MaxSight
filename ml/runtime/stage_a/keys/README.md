# Stage A verification keys (public only)

## Purpose

On-device verification trusts only the Ed25519 **public** keys shipped in the app package. Keys are never
fetched over the network. Policy: `configs/security/artifact_signing.yaml`. Contract:
`docs/contracts/schemas/artifact_signature.json`. Code: `ml/runtime/stage_a/signature_contract.py`.

**Nothing secret may live in this directory** (enforced by `tests/test_stage_a_key_bundle.py`). The private
signing key lives only in the build environment and is passed to the signer by path through
`MAXSIGHT_SIGNING_KEY_FILE`.

> History: until this change the directory held `current.hmac` / `next.hmac`. HMAC verification keys are also
> signing keys, and this repository is public, so those keys must be treated as compromised. They are no
> longer trusted; removing them from git history is an owner decision.

## Trust window

- `current.ed25519.pub`: active verification key (PEM, SubjectPublicKeyInfo)
- `next.ed25519.pub`: rotation candidate accepted during the overlap

An artifact verifies if its `<artifact>.sig.json` was signed by **either** key. With no public key present,
every artifact is refused: devices fail closed until keys are provisioned.

## Provisioning (first time)

1. On a trusted machine or in CI with access to the secret store:
   `python scripts/infra/generate_signing_keypair.py --slot current --private-out /secure/path/maxsight-signing-current.pem`
   (the script refuses a private-key path inside the repository).
2. Store the private PEM in the secret store (e.g. AWS Secrets Manager) and delete the local copy.
3. Commit only `ml/runtime/stage_a/keys/current.ed25519.pub`.
4. In the build: `export MAXSIGHT_SIGNING_KEY_FILE=<mounted secret path>` then
   `python scripts/product/run.py certify ... --sign --artifact <model.pt>`.

## Rotation runbook

1. Generate `next` the same way (`--slot next`); ship `next.ed25519.pub` in an app update while `current`
   remains valid.
2. Switch the build to sign with the `next` private key.
3. After the overlap, promote: rename `next.ed25519.pub` to `current.ed25519.pub` in a later release.
4. Compromised-key revocation requires an app update (app-store latency is an accepted bound).
