#!/usr/bin/env python3
"""Full production rehearsal: run stress scenarios (rain, glare, tilt, combined)..."""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock Flask before importing web_simulator
import unittest.mock
sys.modules['flask'] = unittest.mock.MagicMock()
sys.modules['flask_cors'] = unittest.mock.MagicMock()

import torch
from PIL import Image
import numpy as np

from tools.simulation.web_simulator import MaxSightSimulator
from tools.simulation.config import config
from ml.utils.stress_testing import generate_edge_case_transforms


def get_device(device: str):
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if device == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def pil_from_tensor(x: torch.Tensor) -> Image.Image:
    """Convert [C,H,W] or [1,C,H,W] tensor in [0,1] to PIL Image."""
    if x.dim() == 4:
        x = x.squeeze(0)
    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def run_rehearsal(
    device: str = 'cpu',
    num_frames: int = 5,
    log_dir: Path = None,
    image_size: tuple = (224, 224),
) -> dict:
    log_dir = log_dir or Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'production_rehearsal.log'
    results_file = log_dir / 'production_rehearsal_results.json'

    transforms = generate_edge_case_transforms()
    scenarios = [
        ('baseline', None),
        ('rain_simulation', 'rain_simulation'),
        ('glare_simulation', 'glare_simulation'),
        ('camera_tilt', 'camera_tilt'),
        ('combined_rain_glare', None),  # custom: rain then glare
    ]

    max_alerts = config.max_alerts_per_frame
    breakdowns = []
    alert_counts = []
    errors = []
    start_wall = time.perf_counter()

    with open(log_file, 'w') as lf:
        def log(msg: str):
            print(msg)
            lf.write(msg + '\n')
            lf.flush()

        log(f"Production rehearsal started (device={device}, num_frames={num_frames})")
        log(f"max_alerts_per_frame={max_alerts}, alert_cooldown_frames={config.alert_cooldown_frames}")

        sim = MaxSightSimulator(device=device)
        try:
            for scenario_name, transform_name in scenarios:
                log(f"\n--- Scenario: {scenario_name} ---")
                for frame_idx in range(num_frames):
                    # Build test image
                    base = torch.rand(3, image_size[0], image_size[1])
                    if transform_name and transform_name in transforms:
                        base = transforms[transform_name](base)
                    elif scenario_name == 'combined_rain_glare':
                        base = transforms['rain_simulation'](base)
                        base = transforms['glare_simulation'](base)
                    pil_img = pil_from_tensor(base)

                    t0 = time.perf_counter()
                    try:
                        result = sim.process_frame(pil_img)
                    except Exception as e:
                        errors.append({'scenario': scenario_name, 'frame': frame_idx, 'error': str(e)})
                        log(f"  Frame {frame_idx}: ERROR {e}")
                        continue
                    elapsed = (time.perf_counter() - t0) * 1000

                    dets = result.get('detections', result.get('stats', {}).get('total_detections', 0))
                    n_det = len(dets) if isinstance(dets, list) else int(dets)
                    breakdown = result.get('pipeline_breakdown') or {}
                    total_ms = breakdown.get('total_ms', 0)
                    breakdowns.append({'scenario': scenario_name, 'frame': frame_idx, **breakdown})
                    alert_counts.append(n_det)

                    ok_alerts = 'OK' if n_det <= max_alerts else f'OVER (max={max_alerts})'
                    log(f"  Frame {frame_idx}: detections={n_det} {ok_alerts}, total_ms={total_ms:.1f}, wall_ms={elapsed:.1f}")

            sim.shutdown()
        except Exception as e:
            log(f"Fatal: {e}")
            errors.append({'fatal': str(e)})

    total_wall = time.perf_counter() - start_wall
    over_alerts = sum(1 for c in alert_counts if c > max_alerts)
    summary = {
        'device': device,
        'num_frames_total': len(alert_counts),
        'num_scenarios': len(scenarios),
        'max_alerts_per_frame': max_alerts,
        'over_alert_count': over_alerts,
        'errors': errors,
        'wall_sec': total_wall,
        'breakdowns_sample': breakdowns[:20],
        'alert_counts': alert_counts,
    }
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    with open(log_file, 'a') as lf:
        lf.write(f"\nSummary: {len(alert_counts)} frames, {over_alerts} over-alert frames, {len(errors)} errors, {total_wall:.1f}s wall\n")
        lf.write(f"Results written to {results_file}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Full production rehearsal with stress scenarios')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu',
                        help='Device for inference')
    parser.add_argument('--num-frames', type=int, default=3,
                        help='Frames per scenario')
    parser.add_argument('--log-dir', type=Path, default=Path('logs'),
                        help='Directory for rehearsal log and results JSON')
    args = parser.parse_args()

    summary = run_rehearsal(
        device=args.device,
        num_frames=args.num_frames,
        log_dir=args.log_dir,
    )
    over = summary.get('over_alert_count', 0)
    errs = len(summary.get('errors', []))
    if over > 0 or errs > 0:
        print(f"\nRehearsal finished with {over} over-alert frame(s) and {errs} error(s). Check logs.")
        sys.exit(1)
    print("\nRehearsal passed: all frames within alert budget, no errors.")
    sys.exit(0)


if __name__ == '__main__':
    main()
