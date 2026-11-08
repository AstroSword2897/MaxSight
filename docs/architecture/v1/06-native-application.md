<!-- Part of the MaxSight V1 Technical Architecture Specification. See 00-overview.md. Source: approved plan 2026-10-31. -->

# D6: Native Application Architecture

### D6.1 iOS (Swift 5.10+, SwiftUI, iOS 17+ proposed, Xcode 16; Swift Package Manager)

| Module (target) | Component(s) | Interface (key types) | Depends on | Lifecycle | Tests |
|---|---|---|---|---|---|
| `MaxSightCore` (SPM, no UIKit) | schemas, C-04, C-06, C-07, C-08 logic, C-11 rules, vocabulary | `Tracker.update(_:) -> [TrackedObject]`, `HazardAssessor.assess(...)`, `SafetyController.update(...) -> EnvironmentalState`, `PolicyEngine.decide(...) -> [AssistanceAction]` | none | pure, value types | XCTest + golden vectors (DR-114) |
| `MaxSightPerception` | C-01 adapter, C-02, C-03 (Core ML), C-05 (needs `CameraModel`), `PerceptionActor` | `actor PerceptionActor { func process(_ buf: CVPixelBuffer, meta: FrameMetadata) -> EnvironmentalState }` | Core, CoreML, Vision, AVFoundation | started after ModelStore READY; stopped on background | XCTest with recorded buffers; on-device perf tests |
| `MaxSightOutput` | C-12 | `actor OutputManager { func submit(_ a: AssistanceAction) }` | AVFAudio (`AVSpeechSynthesizer`, `AVAudioSession` .playback/.spokenAudio + duckOthers), CoreHaptics | app lifetime; handles interruption/route-change notifications | XCTest with fake synthesizer; device haptic checks |
| `MaxSightSpeechInput` | C-13 | `func listen() async -> TaskRequest?` | Speech (`SFSpeechRecognizer`, `requiresOnDeviceRecognition=true`), AVAudioEngine | on demand (button/voice trigger) | XCTest |
| `MaxSightNetwork` | C-14, C-15 | `StageBClient.send(_:) async throws -> TaskResponse` (4 s deadline, cancellation via Task), `ConnectivityMonitor` (NWPathMonitor + health ping) | URLSession, Network | on demand / app lifetime | URLProtocol stubs; integration against dev API |
| `MaxSightModelStore` | C-16 | `ModelStore.resolve() -> ModelHandle` (ACTIVE→LKG→REFUSED), `stage(_:)`, `promote(_:)`, `rollback()` | CryptoKit (`Curve25519.Signing` verifies the Ed25519 `<artifact>.sig.json` contract in `docs/contracts/schemas/artifact_signature.json`; originally planned as P256), FileManager (Application Support) | startup; update check | integrity tests (corrupt/unsigned/incompatible) |
| `MaxSightProfile` | C-10, C-17 | `FVPStore`, `ConsentManager.allows(_ class: EgressClass) -> Bool` | Keychain, FileProtection | on change | XCTest |
| `MaxSightDiagnostics` | C-18, C-19 | `Metrics.record(_:)`, `SessionLog.append(_:)` (JSONL, file-protected), `SyncQueue` | OSLog | continuous | XCTest |
| `MaxSightApp` | UI, `AppCoordinator`, onboarding, status indicator | SwiftUI views; `@Observable AppState` | all | app | XCUITest (VoiceOver labels, onboarding flow) |

**Startup** (`AppCoordinator.start()`, DR-032):
1. Load config.
2. Camera permission (`AVCaptureDevice.requestAccess`).
3. `ModelStore.resolve()` (verify signature + checksum + self-check on the bundled `calibration_frame.png`).
4. Build `PerceptionActor`.
5. Init tracker/assessor/controller.
6. Init `OutputManager` + test earcon.
7. Readiness (one frame end to end < 2 s).
8. READY + announce.

**Background:** the camera stops when backgrounded (Q161/164); "Hazard detection paused" is announced on resign-active.

**Info.plist:** `NSCameraUsageDescription`, `NSMicrophoneUsageDescription`, `NSSpeechRecognitionUsageDescription`, `UIBackgroundModes=[audio]` (finish an utterance only). **No** location, motion, LiDAR/depth or other sensor usage (DR-017; DR-210/211 closed for V1). `AVCaptureDevice` is pinned to `.builtInWideAngleCamera`, so no depth data output is configured.

**Model packaging:**
- `Detector.mlpackage` (FP16 default per DR-116 result) + `manifest.json` + `.sig` is bundled as the LKG.
- Updates are staged in `Application Support/models/staging/` and promoted only after verification (DR-034).

### D6.2 Android (DR-215 Open, after the iOS V1)

| Module | Components | Key APIs | Tests |
|---|---|---|---|
| `core` (Kotlin, JVM-only) | port of MaxSightCore | data classes from the same JSON Schemas | JUnit + golden vectors |
| `perception` | C-01…C-05 | CameraX `ImageAnalysis` (`STRATEGY_KEEP_ONLY_LATEST`), LiteRT (`CompiledModel` vs `Interpreter`, benchmark-chosen), YUV→RGB | JUnit + instrumentation |
| `output` | C-12 | `TextToSpeech` (QUEUE_FLUSH for urgency 3), `VibratorManager`/`VibrationEffect.createWaveform`, `AudioFocusRequest` | instrumentation |
| `speech` | C-13 | `SpeechRecognizer` with `EXTRA_PREFER_OFFLINE` | instrumentation |
| `network` | C-14/C-15 | OkHttp (4 s call timeout), `ConnectivityManager.NetworkCallback` | MockWebServer |
| `modelstore` | C-16 | `java.security` ECDSA P-256 verify; app-private storage | JUnit |
| `app` | UI | Jetpack Compose, TalkBack semantics | Espresso |

Android runtime behaviour is **not assumed** identical: separate device benchmarks and parity reports (D8.5).

### D6.3 Repository structure (`[E]` exists · `[P]` planned)

```
ml/runtime/stage_a/          [E] types.py, model_handle.py, verify.py, torch_runner.py (to be replaced), messages.py
   ├─ pipeline.py quality.py [P]   detectors/ tracking/ spatial/ hazard/ controller.py [P]
ml/training/                 [E] train_loop.py, export.py, quantization.py, qat_pipeline.py   + stage_a_train.py [P]
ml/evaluation/               [E] safety_gates.py   + hazard_eval.py, cluster_bootstrap.py, decision_matrix.py [P]
ml/data/                     [E] gold/schema.py, dataset_registry.py   + hazard/ (cvat_import, harmonize, split) [P]
app/ (Python reference)      [E] connectivity/, stage_b/, model_update/, ui/haptic_*   (reference & tests only)
ios/MaxSight/                [P] Xcode project + Packages/{MaxSightCore,…}
android/                     [P] (DR-215)
cloud/stageb/                [P] Lambda source (Python 3.12) + tests
infra/cdk/                   [P] CDK app (Stage B, S3, KMS, Cognito, budgets) ; [E] infra/iam/*.json stubs
docs/architecture/v1/        [P] D1–D10 ; [E] docs/architecture/SYSTEM_ARCHITECTURE.md (superseded sections marked)
docs/adr/                    [P] ; docs/contracts/schemas/stage_a/*.v2.json [P] ; [E] docs/contracts/schemas/*.json (v1)
configs/{stage_a,hazard,camera}/ [P]
tools/simulation/            [E] web simulator (Playwright tests [P])
tests/                       [E] 611 tests ; new tests per D8 [P]
```

### D6.4 Dependency management, configuration, build and release

| Area | Tooling |
|---|---|
| **Python** | `pyproject.toml` `[E]` + `requirements*.txt` `[E]`; add `opencv-python==4.10.0.84`, `effdet==0.4.1`, `motmetrics`, `jsonschema`, `hypothesis` (dev); lock with `pip-tools` → `requirements.lock` [P] |
| **iOS** | SPM for local packages; no third-party runtime deps in V1 |
| **Android** | Gradle version catalog |
| **Cloud** | CDK with a pinned `aws-cdk-lib` |
| **Config** | `configs/stage_a/v1.yaml` (thresholds tagged exp.) is bundled into the model package; app runtime config is `Config.plist` per scheme (Dev/Staging/Prod) holding the API base URL and the Cognito pool ID (**no secrets**) |
| **Build** | CI (GitHub Actions `[E] .github/workflows/ci.yml`, `quality.yml`): lint (ruff/black/isort), types (basedpyright), pytest profiles, schema validation, golden-vector generation; Xcode Cloud or a GitHub macOS runner for the iOS build and tests [P] |
| **Release** | model package (`certification → manifest → sign → s3 releases`) → `ios/` pulls the signed package at build time (script verifies the signature) → TestFlight → App Store. Cloud: `cdk deploy` dev → staging → prod with manual approval |
