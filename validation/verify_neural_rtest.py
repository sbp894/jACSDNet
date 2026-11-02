"""LBHB-internal: jACSDNet v0 reproduces J0 seed 0's neural rtest on the BNT 62-site held-out set.

Needs the LBHB training code and data (neural_auditory_categorization/ACnetTCnet_joint, the BNT p62 panel), so it
runs only inside LBHB. It rebuilds the held-out set exactly as the trainer did (train_joint_megaclf.main:
JointData with the run's site_index, sqrt, snr_thresh 0, Nmin_per_slice 10, num_stims 7812; then
train_joint.build_repeat_datasets = the 4 test repeats spliced end to end) and scores the standalone model's
predicted PSTHs ('mr') with train_joint's own per_cell_corr. The reference is the ORIGINAL checkpoint's stored
metrics (rtest_percell, cell_mean_rtest, site_mean_rtest), not the standalone's copy of them.

    python validation/verify_neural_rtest.py            # CPU; writes validation/results/neural_rtest_v0.json
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOINT = '/auto/users/satya/code/projects_getting_started/neural_auditory_categorization/ACnetTCnet_joint/'
TAG = 'jmc_p62_mega4k12_clfcnn10p128_eps0.3_L0.5_seed0_asb256_bnb256_asg2.985_spe1360_tb30_cratchet_refetch'
SRC_CKPT = JOINT + 'output/checkpoints/' + TAG + '.pt'
OUT_JSON = os.path.join(REPO, 'validation', 'results', 'neural_rtest_v0.json')
TOL_MEAN, TOL_CELL = 1e-4, 1e-3     # the reference was scored on an A40 GPU; this replay runs on CPU

for _p in (JOINT, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import train_joint_megaclf as jmc                          # noqa: E402  (its imports carry v3, tj, jd, jm)
from jacsdnet_model import load_jacsdnet                   # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def git_sha(path):
    sha = subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', path, 'diff', '--quiet', 'HEAD', '--'], capture_output=True).returncode
    return sha + ('-dirty' if dirty else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    device = torch.device(args.device)
    t0 = time.time()
    repo_sha, training_sha = git_sha(REPO), git_sha(JOINT)     # at start: the code that actually ran

    blob = torch.load(SRC_CKPT, map_location='cpu', weights_only=False)
    ref = blob['metrics']
    sites = [int(s) for s in blob['site_index']]
    model, cell_rtest_shipped = load_jacsdnet(map_location=device)
    assert model.provenance['site_index'] == sites, 'standalone and source checkpoints disagree on the sites'

    D = jmc.jd.JointData(site_index=sites, compress=jmc.jm.MATCHED_COMPRESS, snr_thresh=0.0,
                         Nmin_per_slice=jmc.tj.PANEL_NMIN['62'], num_stims=7812,
                         **jmc.v3.frontend_data_kwargs(argparse.Namespace(frontend='gtg')))
    dims = [int(d) for d in D.head_output_dims]
    assert dims == [int(d) for d in blob['site_head_dims']], 'panel site dims differ from the checkpoint'
    names_d = [str(c) for site in D.cell_names for c in np.atleast_1d(site)]
    assert names_d == model.cell_names, 'cell order of the rebuilt panel != the standalone readout order'
    _, test_ds = jmc.tj.build_repeat_datasets(D, device)
    print(f"[data] {D.n_sites} sites, {sum(dims)} cells, test X {tuple(test_ds['X'].shape)} "
          f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    with torch.inference_mode():
        rates = model.readout_from_model_input(test_ds['X'].to(device), 'mr')[0].cpu().float().numpy()
    edges = np.concatenate([[0], np.cumsum(dims)])
    per_cell, per_site = [], {}
    for k in range(len(dims)):
        r = jmc.tj.per_cell_corr(rates[:, edges[k]:edges[k + 1]], test_ds['Y'][k].numpy())
        per_site[k] = float(np.nanmean(r))
        per_cell.append(r)
    per_cell = np.concatenate(per_cell)

    ref_cell = np.asarray(ref['rtest_percell'], np.float64)
    d_cell = np.abs(per_cell - ref_cell)
    res = dict(
        cell_mean=float(np.nanmean(per_cell)), site_mean=float(np.nanmean(list(per_site.values()))),
        ref_cell_mean=float(ref['cell_mean_rtest']), ref_site_mean=float(ref['site_mean_rtest']),
        max_abs_diff_per_cell=float(np.nanmax(d_cell)), median_abs_diff_per_cell=float(np.nanmedian(d_cell)),
        n_cells=int(len(per_cell)), n_nan=int((~np.isfinite(per_cell)).sum()),
        ref_n_nan=int((~np.isfinite(ref_cell)).sum()),
        shipped_cell_rtest_equals_ref=bool(np.allclose(cell_rtest_shipped, ref_cell, equal_nan=True)),
    )
    res['pass'] = bool(abs(res['cell_mean'] - res['ref_cell_mean']) < TOL_MEAN
                       and abs(res['site_mean'] - res['ref_site_mean']) < TOL_MEAN
                       and res['max_abs_diff_per_cell'] < TOL_CELL and res['n_nan'] == res['ref_n_nan']
                       and res['shipped_cell_rtest_equals_ref'])
    res.update(tol_mean=TOL_MEAN, tol_cell=TOL_CELL, device=str(device), host=socket.gethostname().split('.')[0],
               repo_sha=repo_sha, training_code_sha=training_sha, source_checkpoint=SRC_CKPT,
               protocol='train_joint.build_repeat_datasets test split (4 repeats spliced), per_cell_corr; '
                        'site mean = nanmean over the 62 per-site means',
               date=time.strftime('%F %T'))
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(f'[save] {OUT_JSON}')
    print('NEURAL RTEST ' + ('PASSED' if res['pass'] else 'FAILED'))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
