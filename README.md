# jACSDNet

**jACSDNet** (joint Auditory-Cortex / Sound-Detection Net) is one network trained
jointly to (a) predict the PSTHs of 3124 neurons recorded across 62 sites in ferret
(*Mustela putorius furo*) auditory cortex and (b) detect AudioSet sound events
(456 classes). The sound-event classifier reads the **predicted neural rates**, not
the trunk, through a learned 128-d projection.

This is **v0**: the model called *J0* / *joint-AC/SDNet* in the LBHB analyses, training seed 0.

> **Private pre-release, for collaborators only.** Please do not redistribute the code or
> weights. A license has not been chosen yet.

```
waveform
  -> gammatonegram (32 ERB channels, 100 Hz, sqrt)
  -> trunk: 4 causal residual blocks, widths 125-200, kernel 12 (450 ms RF)   'mf'      (T, 200)
  -> 62 site heads (concatenated): Linear + double exponential              'mr'      (T, 3124)
  -> LayerNorm(3124) per bin -> Linear(3124 -> 128, no bias)                'proj'    (T, 128)   <- default
  -> per 500-ms window: CNN10 (2-D; the 128 dims as its frequency axis)     'clf'     (n_win, 512)
  -> Linear(512 -> 456)                                                     'logits'  (n_win, 456)
```

## Install

```bash
conda create -n jacsdnet python=3.11
conda activate jacsdnet
pip install -r requirements.txt
```

Dependencies are `torch`, `torchaudio`, `numpy`, plus `matplotlib` (demo only) and `scipy`
(`nems_match=True` only). The model probes the GPU at load time and falls back to CPU if the
installed wheel has no kernels for it (`best_device()`). For a specific CUDA build, install
`torch`/`torchaudio` from the [PyTorch index](https://pytorch.org) first.

## Quick start

```python
from jacsdnet_model import load_jacsdnet

model, cell_rtest = load_jacsdnet()              # cell_rtest: per-neuron held-out r (3124,)
model.update_audio_process({'overall_db': 60})   # stimulus level in dB SPL (60 is the default)

# Default: the 128-d projection of the predicted rates, every 10 ms
emb, gtg = model.get_embeddings('examples/example_esc50_clip.wav')
#   emb -> (1, time, 128)
#   gtg -> (time, 32)   gammatonegram (model input)

# Any other stage
mf, _     = model.get_embeddings(wav, readout='mf')       # (1, time, 200)   trunk
mr, _     = model.get_embeddings(wav, readout='mr')       # (1, time, 3124)  predicted rates
clf, _    = model.get_embeddings(wav, readout='clf')      # (1, n_win, 512)  classifier embedding
logits, _ = model.get_embeddings(wav, readout='logits')   # (1, n_win, 456)  AudioSet logits

psth = model.predict_psth(wav)            # (time, 3124), spikes/bin at 100 Hz
model.classify(wav, top_k=5)              # per 500-ms window: [(AudioSet label, logit), ...]
model.audioset_labels                     # 456 display names, in logit order
model.cell_names                          # 3124 neuron IDs, in 'mr' column order
```

You can also pass a waveform: `model.get_embeddings(waveform, fs=fs)` with `waveform` a
`(1, n_samples)` tensor. For a batch of gammatonegrams use
`model.readout_from_model_input(x, readout)` with `x` of shape `(batch, time, 32)`.

### Readouts

| readout | shape | time step | what it is |
|---|---|---|---|
| `proj` (default) | (B, T, 128) | 10 ms | LayerNorm over the 3124 predicted rates, then the classifier's learned Linear(3124→128). This is the classifier's own input. |
| `mf` | (B, T, 200) | 10 ms | trunk output ("manifold") |
| `mr` | (B, T, 3124) | 10 ms | predicted PSTHs (spikes/bin), same as `predict_psth` |
| `clf` | (B, n_win, 512) | 500 ms | CNN10 `fc1` (after ReLU), one row per window |
| `logits` | (B, n_win, 456) | 500 ms | AudioSet strong-label logits |

**Windows.** `clf` and `logits` run ONE causal pass over the whole input, then cut it into
consecutive 50-bin (500-ms) windows from bin 0. Each window therefore keeps its full left
context. A trailing remainder shorter than 500 ms is dropped, and an input shorter than 500 ms
raises `ValueError`. For a 500-ms clip that is exactly one window; this is how the ESC-50
numbers below were produced.

**Logits** come from a BCE head trained with a per-class positive weight, so `sigmoid(logit)`
ranks classes but is not a calibrated probability.

## Front end and level

The default front end targets arbitrary wavs at a known level: 60 dB SPL RMS, `level_mode='exact'`,
`lbhb_mode=False` (no peak limiter), sqrt compression. This is the setting every downstream
number below used. Set the level with `model.update_audio_process({'overall_db': ...})`; the
model is level-dependent.

Two things to know:
- During training, the **AudioSet** gammatonegrams were multiplied by 2.985 in the sqrt domain,
  which equals a +19 dB waveform level. The default front end does **not** apply that gain.
  Downstream tests use 60 dB SPL for every model.
- The **BNT** training stimuli came through baphy's path. To reproduce a NEMS recording's `stim`
  from its wavs, use `model.set_bnt_mode(overall_db, fixed_amp_scale)` with the site's own
  exptparams. `model.get_embeddings_from_gtg(stim, gtg_compress='sqrt')` takes a NEMS
  `stim` signal directly; NEMS stims are already sqrt-compressed, the same domain as this model's input.

The front end and the trunk/readout blocks are copied verbatim from the ACNet v1 release.

## Validation

Measured 2026-11-01. Each verdict is a JSON in `validation/results/` and records the git SHA that produced it.

| check | runs | result |
|---|---|---|
| **Parity with the original model** (`validation/check_parity.py`) | anywhere | Six inputs: the example clip, 4 ESC-50 clips, and one 2.37-s input that has 4 classifier windows. The wav→gammatonegram step matches exactly. All five readouts match the original training-code model within 4e-7 relative on CPU. On GPU (RTX 5070 Ti, TF32 off) the readouts are within 4e-6 and the front end within 1.3e-4. |
| **Neural rtest** (`validation/verify_neural_rtest.py`) | LBHB only | BNT 62-site held-out set (4 repeats). Cell-pooled **0.3524**, site-weighted **0.4068**, both equal to the training run's stored values. Max per-cell \|Δr\| is 2.6e-5 over 3124 cells. |
| **ESC-50 d′** (`validation/verify_esc50_dprime.py`) | LBHB only | 2000 500-ms clips at 60 dB SPL, scored with the stored 2-layer MLP probe of each fold. Per-class d′ is identical to the stored value on all 5 folds, with 100% prediction agreement. Mean over folds: **MR 2.7163** (top 1562 units by rtest), **CLF 2.8907**. |

ESC-50 per-fold d′ (test fold 1–5):

| readout | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean |
|---|---|---|---|---|---|---|
| MR | 2.7730 | 2.5144 | 2.7523 | 2.9935 | 2.5481 | 2.7163 |
| CLF | 2.9257 | 2.8538 | 2.8452 | 3.0540 | 2.7745 | 2.8907 |

Notes on the validation:
- **ESC-50 embeddings differ from the stored dumps by 2e-4 relative.** The stored dumps were computed on a GPU. The original model run on CPU shows the same 2.2e-4 (MR) / 2.7e-4 (CLF) gap, and the standalone matches the original on CPU to 1e-7 over all 2000 clips (`validation/claude_debug/`).
- **The default 128-d `proj` readout has no stored benchmark of its own.** It is validated by parity, and because `clf` is computed from it.
- **Stored only, not re-run here:** AudioSet mAP (test 0.1477, val 0.1207).

```bash
python validation/check_parity.py                  # anywhere, seconds
python validation/check_parity.py --device cuda
```

## Demo

```bash
python demo_jacsdnet.py
```

Edit the settings block at the top to change the wav (`WAV_FILE`) or level (`DBSPL`). The demo saves
`examples/<wav name>.png` (300 dpi) with four parts: the gammatonegram, the 128-d embedding,
the predicted PSTHs sorted by rtest, and the top AudioSet classes per window.

## Files

| File | Purpose |
|------|---------|
| `jacsdnet_model.py` | Model, audio front end, `load_jacsdnet()`. |
| `gtgram.py`, `gtg_filters.py` | Self-contained gammatonegram front end (from ACNet v1). |
| `demo_jacsdnet.py` | Runnable demo. |
| `weights/jacsdnet_v0.pt` | Weights + config + `cell_rtest`, `cell_names`, AudioSet labels, provenance (`model.provenance`). |
| `examples/example_esc50_clip.wav` | Example clip (ESC-50, CC BY-NC). |
| `validation/` | Parity fixture + checks (see above); `results/*.json` are the verdicts. |
| `data/convert_j0_to_standalone.py` | How `weights/jacsdnet_v0.pt` was made from the training checkpoint (LBHB only). |

## Notes

- **Version.** `model.version == 'v0'`, stored in the checkpoint. It is training seed 0 of J0.
  Seeds 1 and 2 exist but are not exported. Their stored rtest is 0.3555 / 0.3539 cell-pooled.
- **Units.** PSTHs are spikes per 10-ms bin, matching the training targets.
- **GPU numerics.** For bit-level comparisons on Ampere or newer GPUs, turn TF32 off
  (`torch.backends.cudnn.allow_tf32 = False`).

Copyright (c) 2026 Satyabrata Parida and Stephen V. David. License: not yet chosen (private pre-release).
