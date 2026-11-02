"""debug_stored_dump_numerics.py -- is the 2e-4 gap between jACSDNet and the stored ESC-50 dumps the dumps' own numerics?

The parity fixture holds the ORIGINAL training-code model's outputs, computed on CPU, for the first 4 ESC50_500ms_v0
clips (the same rows 0..3 of the stored dumps). If original-on-CPU vs stored differs by the same ~2e-4, the gap is
how the dumps were computed (GPU), not the standalone. Also prints whatever host/device fields the dumps record.
"""
import os
import sys

import numpy as np

NAC = '/auto/users/satya/code/projects_getting_started/neural_auditory_categorization/'
sys.path.insert(0, NAC + 'ESC50')
import tcmega_joint_reps as TJ          # noqa: E402
import esc50_xmodel_registry as XR      # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fx = np.load(os.path.join(REPO, 'validation', 'parity_fixture.npz'))
names = list(fx['names'])
for ro, key, ek in (('mr', 'jmc_eps0.3_cnn10_arc_MR', 'meanrates'), ('clf', 'jmc_eps0.3_cnn10_arc_CLF', 'mf')):
    spec = TJ.TCMEGA_JOINT_MODELS[key]
    d = XR._CpuUnpickler(open(spec['emb_pkl'], 'rb')).load()
    files = list(d['filenames'])
    ref = np.asarray(d[ek], np.float64)
    scale = np.abs(ref).max()
    print(f'== {key}: {spec["emb_pkl"]}')
    print('   recorded:', {k: d.get(k) for k in ('host', 'device', 'git_sha', 'source', 'tag') if k in d})
    for i, n in enumerate(names):
        if n not in files:
            continue
        j = files.index(n)
        orig = fx[f'{i}_{ro}'][0]
        orig = orig.mean(0) if ro == 'mr' else orig.mean(0)            # mr: time-mean; clf: mean over windows (1)
        print(f'   {n:28s} original-CPU vs stored: max|d|/max|stored| = {np.abs(orig - ref[j]).max() / scale:.2e}')
