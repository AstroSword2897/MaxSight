#!/usr/bin/env python3
"""Canonical product pipeline runner. Dispatches to train, validate, export, package, smoke per docs/productization/03_pipeline_declutter_map.md."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _run(cmd: list, env=None) -> int:
    return subprocess.run(cmd, cwd=REPO, env=env).returncode


def cmd_train(args) -> int:
    script = REPO / "scripts" / "train_maxsight.py"
    if not script.exists():
        print("train_maxsight.py not found", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script)]
    if getattr(args, "data_dir", None):
        cmd += ["--data-dir", args.data_dir]
    if getattr(args, "checkpoint_dir", None):
        cmd += ["--checkpoint-dir", args.checkpoint_dir]
    if getattr(args, "epochs", None) is not None:
        cmd += ["--epochs", str(args.epochs)]
    if getattr(args, "batch_size", None) is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if getattr(args, "device", None):
        cmd += ["--device", args.device]
    cmd += args.extra or []
    return _run(cmd)


def cmd_validate(args) -> int:
    # Run test suite first.
    pytest_args = ["tests/", "-v", "--tb=short"]
    if getattr(args, "skip_export_tests", False):
        pytest_args += ["--ignore=tests/test_export_validation.py"]
    if getattr(args, "no_x", False):
        pass
    else:
        pytest_args += ["-x"]
    code = _run([sys.executable, "-m", "pytest"] + pytest_args)
    if code != 0:
        return code
    if getattr(args, "data", False):
        script = REPO / "scripts" / "validate_data_pipeline.py"
        if script.exists():
            code = _run([sys.executable, str(script)])
            if code != 0:
                return code
    if getattr(args, "checkpoint", None):
        ckpt = Path(args.checkpoint)
        if ckpt.exists():
            # Quick load + forward pass.
            sys.path.insert(0, str(REPO))
            import torch
            from ml.models.maxsight_cnn import create_model
            model = create_model()
            ckpt_data = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            state = ckpt_data.get("model_state_dict", ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data
            model.load_state_dict(state, strict=False)
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            print("Checkpoint load and forward pass OK.")
        else:
            print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
            return 1
    return 0


def cmd_export(args) -> int:
    cmd = [sys.executable, "-m", "ml.training.export", "--device", "cpu"]
    if getattr(args, "checkpoint", None):
        cmd += ["--checkpoint", args.checkpoint]
    if getattr(args, "format", None):
        cmd += ["--format", args.format]
    if getattr(args, "output", None):
        cmd += ["--output", args.output]
    return _run(cmd)


def cmd_package(args) -> int:
    script = REPO / "scripts" / "export_for_xcode.py"
    if not script.exists():
        print("export_for_xcode.py not found", file=sys.stderr)
        return 1
    ckpt = getattr(args, "checkpoint", None) or "checkpoints/final_model.pt"
    out = getattr(args, "output", None) or "maxsight_ios_bundle"
    return _run([sys.executable, str(script), ckpt, out])


def cmd_smoke(args) -> int:
    script = REPO / "scripts" / "smoke_train.py"
    if not script.exists():
        print("smoke_train.py not found", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script), "--force-cpu"]
    if getattr(args, "epochs", None) is not None:
        cmd += ["--epochs", str(args.epochs)]
    code = _run(cmd)
    if code != 0:
        return code
    # Quick inference smoke: create model and one forward.
    sys.path.insert(0, str(REPO))
    import torch
    from ml.models.maxsight_cnn import create_model
    model = create_model()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(x)
    print("Smoke: training + inference OK.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="MaxSight canonical product pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train production model")
    p_train.add_argument("--data-dir", required=True, help="Data root")
    p_train.add_argument("--checkpoint-dir", default="./checkpoints")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("extra", nargs="*", help="Extra args for train_maxsight.py")

    # validate
    p_val = sub.add_parser("validate", help="Run tests and optional checkpoint/data checks")
    p_val.add_argument("--checkpoint", help="Optional checkpoint to load and run one forward")
    p_val.add_argument("--data", action="store_true", help="Run validate_data_pipeline.py")
    p_val.add_argument("--skip-export-tests", action="store_true", help="Skip test_export_validation (JIT trace can be flaky per docs/status.md)")

    # export
    p_exp = sub.add_parser("export", help="Export checkpoint to CoreML/JIT/ONNX/ExecuTorch")
    p_exp.add_argument("--checkpoint", required=True)
    p_exp.add_argument("--format", choices=["jit", "coreml", "onnx", "executorch"], default="coreml")
    p_exp.add_argument("--output", required=True)

    # package
    p_pkg = sub.add_parser("package", help="Build Xcode-ready bundle")
    p_pkg.add_argument("--checkpoint", default="checkpoints/final_model.pt")
    p_pkg.add_argument("--output", default="maxsight_ios_bundle")

    # smoke
    p_smoke = sub.add_parser("smoke", help="Short training + inference sanity")
    p_smoke.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()
    if args.command == "train":
        return cmd_train(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "export":
        return cmd_export(args)
    if args.command == "package":
        return cmd_package(args)
    if args.command == "smoke":
        return cmd_smoke(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
