"""
Demo: load jACSDNet, run a wav file, and plot the gammatonegram, the default
128-d embedding (the classifier's projection of the predicted rates), the
predicted neural PSTHs, and the top AudioSet classes per 500-ms window.

Edit the settings block below and run:  python demo_jacsdnet.py
The figure is saved next to the wav file, with the same base name (300 dpi).
"""

import os
import subprocess

# Import torch first. In some MKL/OpenMP setups, importing numpy (or anything
# that pulls it in, like matplotlib) before torch deadlocks `import torch`.
import torch
import numpy as np
from jacsdnet_model import load_jacsdnet, best_device

import matplotlib
# matplotlib.use('Agg')  # uncomment for headless machines (no display); saves only
import matplotlib.pyplot as plt

# ---------------------------- settings ------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))

WAV_FILE = os.path.join(HERE, 'examples', 'example_esc50_clip.wav')  # audio to run
DBSPL = 60.0            # stimulus level in dB SPL (the model is level-dependent)
WEIGHTS_PATH = None     # None -> weights/jacsdnet_v0.pt
TOP_K = 3               # AudioSet classes listed per 500-ms window
# --------------------------------------------------------------------------- #

OUT_PNG = os.path.splitext(WAV_FILE)[0] + '.png'
try:
    GIT_SHA = subprocess.run(['git', '-C', HERE, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True).stdout.strip() or 'unknown'
except OSError:
    GIT_SHA = 'unknown'

device = best_device()
model, cell_rtest = load_jacsdnet(WEIGHTS_PATH)
model.update_audio_process({'overall_db': DBSPL, 'level_mode': 'exact', 'lbhb_mode': False})
model.to(device)
model.eval()

with torch.no_grad():
    proj, gtg = model.get_embeddings(WAV_FILE)                    # default readout: 128-d projection
    psth = model.predict_psth(WAV_FILE).cpu().numpy()
    n_bins = gtg.shape[0]
    top = model.classify(WAV_FILE, top_k=TOP_K) if n_bins >= model.config['window_bins'] else []

proj = proj.cpu().numpy().squeeze(0)                              # (time, 128)
gtg = gtg.cpu().numpy()                                           # (time, 32)
cf_kHz = model.audio_process.get_cfs() / 1e3
fs_gtg = model.audio_process.fs_gtg
dur_ms = 1e3 * n_bins / fs_gtg
print(f'gammatonegram {gtg.shape}, projection {proj.shape}, PSTH {psth.shape} [time x dim], fs={fs_gtg} Hz')
for w, row in enumerate(top):
    print(f'window {w} ({500 * w}-{500 * (w + 1)} ms): ' + ', '.join(f'{lab} ({logit:+.2f})' for lab, logit in row))

fig, ax = plt.subplots(3, 1, figsize=(7, 6.6), sharex=True)
ax[0].imshow(gtg.T, origin='lower', aspect='auto', extent=(0, dur_ms, 0, gtg.shape[1]))
cf_tick_kHz = np.array([0.2, 2, 20])
cf_tick_pos = [int(np.argmin(np.abs(cf_kHz - t))) + 0.5 for t in cf_tick_kHz]
ax[0].set_yticks(cf_tick_pos)
ax[0].set_yticklabels([f'{t:g}' for t in cf_tick_kHz])
ax[0].set(ylabel='CF (kHz)', title='Gammatonegram')

ax[1].imshow(proj.T, origin='lower', aspect='auto', extent=(0, dur_ms, 0, proj.shape[1]))
ax[1].set(ylabel='projection dim', title='jACSDNet embedding (128-d projection of predicted rates)')

order = np.argsort(-cell_rtest) if cell_rtest is not None else np.arange(psth.shape[1])
ax[2].imshow(psth[:, order].T, origin='lower', aspect='auto', extent=(0, dur_ms, 0, psth.shape[1]))
ax[2].set(ylabel='neuron (sorted by r_test)', title='Predicted neural PSTHs', xlabel='time (ms)')
for w, row in enumerate(top):
    ax[0].text(500 * w + 10, gtg.shape[1] - 1, '\n'.join(lab for lab, _ in row), va='top', ha='left',
               fontsize=6, color='w')

fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.text(0.99, 0.005, f'jACSDNet {model.version} | git {GIT_SHA} | {os.path.basename(WAV_FILE)} @ {DBSPL:g} dB SPL',
         ha='right', va='bottom', fontsize=6, color='0.4')
fig.savefig(OUT_PNG, dpi=300)
print(f'Saved figure to {OUT_PNG}')
