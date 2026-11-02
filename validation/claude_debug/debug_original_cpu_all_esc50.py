"""debug_original_cpu_all_esc50.py -- where does the ~2e-4 standalone-vs-stored ESC-50 embedding gap come from?

Runs the ORIGINAL training-code model (train_joint_megaclf.load_joint_megaclf) with the ORIGINAL front end (the acnetview's
audio_process, torchaudio.load), exactly as jmc_readouts / esc50_mr_embeddings do, but on CPU, over all 2000 ESC50_500ms_v0
clips, and compares MR (time-mean rates) and CLF (fc1, jmc_readouts.window_features) three ways:
  original-CPU vs standalone-CPU, original-CPU vs stored dump, standalone-CPU vs stored dump.
Also locates the worst clip/column of each gap.
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAC = '/auto/users/satya/code/projects_getting_started/neural_auditory_categorization/'
JOINT = NAC + 'ACnetTCnet_joint/'
TAG = 'jmc_p62_mega4k12_clfcnn10p128_eps0.3_L0.5_seed0_asb256_bnb256_asg2.985_spe1360_tb30_cratchet_refetch'
for _p in (JOINT, NAC + 'TCnet_mega', NAC + 'ESC50', REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torchaudio                               # noqa: E402
import train_joint_megaclf as jmc               # noqa: E402
import jmc_readouts as jr                       # noqa: E402
import esc50_mr_embeddings as eme               # noqa: E402
import tcmega_joint_reps as TJ                  # noqa: E402
import esc50_xmodel_registry as XR              # noqa: E402
from jacsdnet_model import load_jacsdnet        # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device('cpu')

view, *_ = eme._build_joint(dict(ckpt=JOINT + 'output/checkpoints/' + TAG + '_acnetview.pt', note=''), dev)
view.params_audio_proc['model_mode'] = 'audio'
view.update_audio_process(params_audio_proc={'overall_db': eme.DBSPL, 'level_mode': eme.LEVEL_MODE,
                                             'lbhb_mode': eme.LBHB_MODE})
view.audio_process.device = dev
view.eval()
joint, _ = jmc.load_joint_megaclf(JOINT + 'output/checkpoints/' + TAG + '.pt', dev)
sa, _ = load_jacsdnet(map_location='cpu')

files = sorted(f for f in os.listdir(eme.WAV_DIR) if f.endswith('.wav'))
with torch.no_grad():
    g_orig = torch.stack([view.audio_process(*torchaudio.load(eme.WAV_DIR + f)).squeeze().cpu() for f in files])
    g_sa = torch.stack([sa._to_gtg(eme.WAV_DIR + f) for f in files])
    mr_o, clf_o, mr_s, clf_s = [], [], [], []
    for i in range(0, len(files), 100):
        wf = jr.window_features(joint, g_orig[i:i + 100])
        mr_o.append(wf['meanrate']); clf_o.append(wf['clf_flatten'].mean(1))
        mr_s.append(sa.readout_from_model_input(g_sa[i:i + 100], 'mr').mean(1).numpy())
        clf_s.append(sa.readout_from_model_input(g_sa[i:i + 100], 'clf').mean(1).numpy())
mr_o, clf_o, mr_s, clf_s = map(np.concatenate, (mr_o, clf_o, mr_s, clf_s))
print(f'gtg original-CPU vs standalone-CPU over {len(files)} clips: '
      f'{np.abs(g_orig.numpy() - g_sa.numpy()).max() / np.abs(g_orig.numpy()).max():.2e}')


def gap(a, b, label, names):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    per_clip = d.max(1) / np.abs(b).max()
    print(f'  {label:32s} rel max {d.max() / np.abs(b).max():.2e} | worst clip {names[i]} col {j} '
          f'({a[i, j]:.5f} vs {b[i, j]:.5f}) | clips > 1e-4: {(per_clip > 1e-4).sum()}  > 1e-5: {(per_clip > 1e-5).sum()}')


for ro, key, ek, o, s in (('MR', 'jmc_eps0.3_cnn10_arc_MR', 'meanrates', mr_o, mr_s),
                          ('CLF', 'jmc_eps0.3_cnn10_arc_CLF', 'mf', clf_o, clf_s)):
    d = XR._CpuUnpickler(open(TJ.TCMEGA_JOINT_MODELS[key]['emb_pkl'], 'rb')).load()
    assert list(d['filenames']) == files
    st = np.asarray(d[ek], np.float32)
    print(f'== {ro}')
    gap(o, s, 'original-CPU vs standalone-CPU', files)
    gap(o, st, 'original-CPU vs stored dump', files)
    gap(s, st, 'standalone-CPU vs stored dump', files)
