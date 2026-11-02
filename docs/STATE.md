# State — jACSDNet_v0

Last updated 2026-11-01.

## Done (v0)
- Standalone package built from J0 seed 0
  (`ACnetTCnet_joint/output/checkpoints/jmc_p62_mega4k12_clfcnn10p128_eps0.3_L0.5_seed0_..._cratchet_refetch.pt`,
  trained at `061f5fbd` on ARC). Default readout = 128-d projection.
- Validation, all passing (verdicts in `validation/results/`):
  - **parity with the original model:** 6 inputs x 5 readouts, front end exact, worst 4e-7 on CPU;
    on GPU (TF32 off) readouts 4e-6 and front end 1.3e-4;
  - **neural rtest:** 0.3524 / 0.4068, the run's stored values, max per-cell |Δ| 2.6e-5;
  - **ESC-50 d′:** per-fold identical to the stored probes on all 5 folds (MR 2.7163, CLF 2.8907).
- Pushed to the private `sbp894/jACSDNet`, tag `v0`.

## Open
- [ ] **License.** None chosen yet; the README says private pre-release. Satya's call.
- [ ] **Older torch (pt_nems, torch 2.4) not tested.** manatee and capybara both refused launches 2026-11-01
      (MATLAB running). Run `validation/check_parity.py` there when they free up.
- [ ] **No stored benchmark for the default `proj` readout.** ESC-50 d′ of the 128-d projection was never computed
      in the analyses. It is validated by parity and as the input of `clf`.
- [ ] **Seeds 1 and 2 not exported.** If wanted, export them as new version files, not overwrites.
- [ ] **Collaborator access** is managed on GitHub. On a personal-account repo, collaborators get push access; there is no read-only role.
