# CLAUDE.md — jACSDNet_v0

Standalone, dependency-light release of **J0 (joint-AC/SDNet), seed 0**: waveform → 128-d projection of
predicted rates (default), trunk manifold, 3124 predicted PSTHs, CNN10 classifier embedding, AudioSet logits.
Patterned on `../ACNet_v1/` (front end and trunk blocks copied verbatim from its `acnet_model.py`).

**A separate, PRIVATE git repo** (`git@github.com:sbp894/jACSDNet.git`, created by Satya 2026-11-01), nested in
`projects_getting_started` and ignored by the parent (`.gitignore`). Commits here do not go to the parent repo.
Collaborators only; no license chosen yet.

- **Push only with Satya's go for that push.** He authorized the v0 build + push on 2026-11-01. Never force-push.
- One branch, `main`. Tag `v0` marks the release.
- Two Claude accounts share this machine: see `../pytorch_models/docs/COEXIST.md`.

## Prove it still works

```bash
python validation/check_parity.py               # anywhere: standalone == original J0 on 6 inputs x 5 readouts
python validation/verify_neural_rtest.py        # LBHB only: BNT p62 held-out rtest == the run's stored 0.3524 / 0.4068
python validation/verify_esc50_dprime.py        # LBHB only: ESC-50 per-fold d' == stored (MR 2.7163, CLF 2.8907)
```

Each exits 1 on failure; the LBHB ones write `validation/results/*.json` with the SHA they ran at. Run them through
the `launch-ssh-job` skill (rhino, CPU is fine). Say "`check_parity` passes", not "tests pass".

## Hard constraints

- **Dependencies are torch, torchaudio, numpy (+ matplotlib for the demo, scipy for nems_match).** No sppy, no
  `PT_EncMdl_helpers_v2`, no sklearn in `jacsdnet_model.py`. Only `data/` and the `validation/verify_*` scripts may
  import LBHB code, and they say so.
- **`weights/jacsdnet_v0.pt` is made only by `data/convert_j0_to_standalone.py`**, which asserts parity against the
  original model before saving and regenerates `validation/parity_fixture.npz` with it. Never hand-edit either.
  A new export (another seed, a retrain) is a new version file (`jacsdnet_v1.pt`), not an overwrite — ask first.
- The default readout is `'proj'` (Satya 2026-11-01: "I want access to the 128d projection from MR. Make that the
  default"). Keep it the default.
- `clf` / `logits` = one causal pass, consecutive 50-bin windows from bin 0, remainder dropped — the
  `jmc_readouts.window_features` rule the stored ESC-50 / GISE51 numbers used. Changing it breaks comparability.

## Sharp edges

- The stored ESC-50 dumps differ from ANY CPU run of J0 by ~2e-4 relative (they were made on a GPU); the
  standalone matches the original model on CPU to 1e-7. `validation/claude_debug/debug_original_cpu_all_esc50.py`.
- GPU vs CPU front end: 1.3e-4 relative (float32 filterbank arithmetic); readouts 4e-6 with TF32 off.
- AudioSet training inputs carried `as_gain` 2.985 (sqrt domain, = +19 dB); downstream readouts use 60 dB SPL and
  no gain. Do not "fix" the default level to 79 dB.

Provenance: anything here that writes a figure / table / derived file records `git rev-parse --short HEAD`.
`docs/STATE.md` = what is open.
