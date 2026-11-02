"""Portable check: the standalone jACSDNet reproduces the original J0 model's outputs.

Needs only this repo (no LBHB code or data). `validation/parity_fixture.npz` holds, for the example clip and
4 ESC-50 500-ms clips, the ORIGINAL training-code model's gammatonegram and its five readouts (mf, mr, proj,
clf, logits), written by data/convert_j0_to_standalone.py. This script recomputes them with jacsdnet_model.py
and exits 1 on any mismatch.

    python validation/check_parity.py              # CPU (exact: tolerance 1e-5 relative)
    python validation/check_parity.py --device cuda   # GPU (TF32 off; tolerance 1e-4 relative)
"""
import argparse
import os
import sys

import torch
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from jacsdnet_model import load_jacsdnet, READOUTS          # noqa: E402

FIXTURE = os.path.join(REPO, 'validation', 'parity_fixture.npz')
EXAMPLE_WAV = os.path.join(REPO, 'examples', 'example_esc50_clip.wav')


def rel_max(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tol = 1e-5 if args.device == 'cpu' else 1e-4

    model, _ = load_jacsdnet(map_location=args.device)
    fx = np.load(FIXTURE)
    names = list(fx['names'])
    worst, failed = 0.0, []
    with torch.no_grad():
        for i, name in enumerate(names):
            row = {}
            if i == 0:                                     # the one wav shipped here: check the front end too
                gtg = model._to_gtg(EXAMPLE_WAV).cpu().numpy()
                row['gtg'] = rel_max(gtg, fx[f'{i}_gtg'])
            x = torch.from_numpy(fx[f'{i}_gtg']).to(args.device)
            for ro in READOUTS:
                row[ro] = rel_max(model.readout_from_model_input(x, ro).cpu().numpy(), fx[f'{i}_{ro}'])
            bad = [k for k, v in row.items() if not v < tol]
            failed += [f'{name}:{k}' for k in bad]
            worst = max(worst, max(row.values()))
            print(f'{name:32s} ' + ' '.join(f'{k}={v:.1e}' for k, v in row.items()) + ('  FAIL' if bad else ''))
    if failed:
        print(f'PARITY FAILED (tol {tol:g}): {failed}')
        sys.exit(1)
    print(f'PARITY PASSED: worst rel max|d| {worst:.1e} (tol {tol:g}) on {len(names)} inputs x {len(READOUTS)} readouts')


if __name__ == '__main__':
    main()
