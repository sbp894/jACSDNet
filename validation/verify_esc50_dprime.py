"""LBHB-internal: jACSDNet v0 reproduces J0 seed 0's ESC-50 embeddings and per-class d' (500-ms clips, 60 dB SPL).

Needs the LBHB analysis code and the ESC-50 500-ms clip set, so it runs only inside LBHB.

1. Embed every ESC50_500ms_v0 clip (2000) with the standalone, wav -> its own front end (60 dB SPL, exact,
   lbhb_mode False):  MR = time-mean of readout 'mr' (3124);  CLF = readout 'clf' (one 500-ms window, 512).
2. Embedding parity against the stored dumps the published numbers were computed from
   (tcmega_joint_reps.TCMEGA_JOINT_MODELS['jmc_eps0.3_cnn10_arc_MR' / '_CLF'] emb_pkl).
3. d' replay, per fold: the STORED trained 2-layer MLP probe of that fold (clf_pkl accuracy_data_emb
   model_classifier[jk]) applied to the standalone's embeddings, with esc50_xmodel_classify's own preprocessing
   (MR: top-50% rtest unit pool, best first, first n_best; z-score from that fold's training rows). The probe is
   held fixed, so the only thing that can move d' is the embedding. Per-class d' = esc50_xmodel_registry's.
   The same replay on the stored embeddings must give the stored predictions exactly (checks the replay itself).

    python validation/verify_esc50_dprime.py                # all 5 folds; writes validation/results/esc50_dprime_v0.json
    python validation/verify_esc50_dprime.py --folds 0      # one fold
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
NAC = '/auto/users/satya/code/projects_getting_started/neural_auditory_categorization/'
JOINT = NAC + 'ACnetTCnet_joint/'
ESC50_DIR = '/auto/users/satya/code/projects_getting_started/downloaded/dataset/ESC-50-master/misc_out/ESC50_500ms_v0/'
KEY = 'jmc_eps0.3_cnn10_arc'          # J0 seed 0 in the ESC-50 registry
OUT_JSON = os.path.join(REPO, 'validation', 'results', 'esc50_dprime_v0.json')
# The stored dumps were computed on a GPU: the ORIGINAL training-code model, run on CPU over all 2000 clips, differs
# from them by 2.16e-4 (MR) / 2.71e-4 (CLF) relative max -- exactly the standalone's gap -- while standalone vs
# original-CPU is 1.2e-7 / 4.3e-7 and the gammatonegrams are identical (validation/claude_debug/
# debug_original_cpu_all_esc50.py, 2026-11-01). So the embedding gate is 1e-3, and the d' / prediction gates carry
# the weight.
TOL_EMB_REL = 1e-3
TOL_DPRIME = 1e-6
DBSPL = 60.0

for _p in (JOINT, NAC + 'ESC50', REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import esc50_xmodel_classify as XC                        # noqa: E402  (load_labels, make_splits, zscore_folds, probe)
import esc50_xmodel_registry as XR                        # noqa: E402  (per_class_dprime, _CpuUnpickler)
import tcmega_joint_reps as TJ                            # noqa: E402  (emb_pkl / clf_pkl of each registry key)
from jacsdnet_model import load_jacsdnet                  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def git_sha(path):
    sha = subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', path, 'diff', '--quiet', 'HEAD', '--'], capture_output=True).returncode
    return sha + ('-dirty' if dirty else '')


def load_pkl(path):
    with open(path, 'rb') as fh:
        return XR._CpuUnpickler(fh).load()


def rel_max(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def probe_preds(arr, d_emb, d_clf, readout, splits, y, jk, device):
    """esc50_xmodel_classify.main + run_arm, final pass, fold jk, with the stored probe instead of a retrained one."""
    cols, sl = None, None
    if readout == 'mr':
        rtest = np.asarray(d_emb['rtest_percell'], dtype=np.float64)
        thresh = float(np.nanpercentile(rtest, 100 - XC.MR_SNR_PCT))
        pool = np.where(rtest >= thresh)[0]
        cols = pool[np.argsort(rtest[pool])[::-1]]
        assert len(cols) == d_clf['n_pool'], (len(cols), d_clf['n_pool'])
        sl = slice(0, int(d_clf['n_best']))
    a = np.asarray(arr)[:, cols] if cols is not None else np.asarray(arr)
    X = XC.zscore_folds(a, splits, device)[jk]
    X = X if sl is None else X[:, sl]
    ad = d_clf['accuracy_data_emb']
    clf = XC.build_classifier(d_clf['classifier_optim_params'], device)
    clf.load_state_dict(ad['model_classifier'][jk])
    clf.eval()
    te = splits[jk][2]
    with torch.no_grad():
        pred, acc = XC.test_model(clf, X[te], y[te])
    return np.asarray(pred).ravel(), float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    folds_req = [int(f) for f in args.folds.split(',')]
    device = torch.device(args.device)
    t0 = time.time()
    repo_sha, analysis_sha = git_sha(REPO), git_sha(JOINT)     # at start: the code that actually ran

    model, _ = load_jacsdnet(map_location=device)
    model.update_audio_process({'overall_db': DBSPL, 'level_mode': 'exact', 'lbhb_mode': False})
    files = sorted(f for f in os.listdir(ESC50_DIR) if f.endswith('.wav'))
    gtg = []
    with torch.no_grad():
        for f in files:
            gtg.append(model._to_gtg(ESC50_DIR + f))
    gtg = torch.stack(gtg)                                 # (2000, 50, 32)
    with torch.no_grad():
        new = {'mr': torch.cat([model.readout_from_model_input(gtg[i:i + 100], 'mr').mean(1)
                                for i in range(0, len(files), 100)]).cpu().numpy(),
               'clf': torch.cat([model.readout_from_model_input(gtg[i:i + 100], 'clf').mean(1)
                                 for i in range(0, len(files), 100)]).cpu().numpy()}
    print(f"[embed] {len(files)} clips, gtg {tuple(gtg.shape)} | MR {new['mr'].shape} CLF {new['clf'].shape} "
          f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    y_np, fold_ids = XC.load_labels(files)
    y = torch.tensor(y_np, dtype=torch.long, device=device)
    splits = XC.make_splits(fold_ids)
    res = dict(n_clips=len(files), folds=folds_req, readouts={})
    ok = True
    for readout, spec_key, emb_key in (('mr', f'{KEY}_MR', 'meanrates'), ('clf', f'{KEY}_CLF', 'mf')):
        spec = TJ.TCMEGA_JOINT_MODELS[spec_key]
        d_emb, d_clf = load_pkl(spec['emb_pkl']), load_pkl(spec['clf_pkl'])
        assert list(d_emb['filenames']) == files, f'{spec_key}: stored dump has a different clip list'
        ref_emb = np.asarray(d_emb[emb_key], np.float32)
        r_emb = rel_max(new[readout], ref_emb)
        ad = d_clf['accuracy_data_emb']
        rows = []
        for jk in folds_req:
            te = splits[jk][2]
            true_te = y_np[te]
            assert np.array_equal(np.asarray(ad['true_test'][jk]).ravel(), true_te), f'fold {jk} test labels differ'
            pred_stored = np.asarray(ad['pred_test'][jk]).ravel()
            dp_stored = XR.per_class_dprime(pred_stored, true_te)
            pred_replay, _ = probe_preds(ref_emb, d_emb if readout == 'mr' else None, d_clf, readout,
                                         splits, y, jk, device)
            pred_new, acc_new = probe_preds(new[readout], d_emb if readout == 'mr' else None, d_clf, readout,
                                            splits, y, jk, device)
            row = dict(fold_index=jk, test_fold=int(fold_ids[te][0]), n_test=int(len(te)),
                       dprime_stored=dp_stored, dprime_standalone=XR.per_class_dprime(pred_new, true_te),
                       acc_stored=float(np.mean(pred_stored == true_te)), acc_standalone=float(np.mean(pred_new == true_te)),
                       replay_of_stored_matches=bool(np.array_equal(pred_replay, pred_stored)),
                       pred_agreement=float(np.mean(pred_new == pred_stored)))
            rows.append(row)
            print(f"[{readout}] fold {jk} (test fold {row['test_fold']}, n={row['n_test']}): d' stored "
                  f"{row['dprime_stored']:.4f} standalone {row['dprime_standalone']:.4f} | acc "
                  f"{row['acc_stored']:.4f} / {row['acc_standalone']:.4f} | pred agreement {row['pred_agreement']:.4f} "
                  f"| replay-of-stored exact {row['replay_of_stored_matches']}", flush=True)
        out = dict(emb_pkl=spec['emb_pkl'], clf_pkl=spec['clf_pkl'], emb_rel_max_diff=r_emb,
                   n_best=d_clf.get('n_best'), n_pool=d_clf.get('n_pool'), folds=rows,
                   dprime_stored_mean=float(np.mean([r['dprime_stored'] for r in rows])),
                   dprime_standalone_mean=float(np.mean([r['dprime_standalone'] for r in rows])))
        out['pass'] = bool(r_emb < TOL_EMB_REL and all(r['replay_of_stored_matches'] for r in rows)
                           and all(abs(r['dprime_standalone'] - r['dprime_stored']) < TOL_DPRIME for r in rows))
        ok &= out['pass']
        res['readouts'][readout] = out
        print(f"[{readout}] embeddings rel max|d| {r_emb:.2e} | mean d' over folds {folds_req}: stored "
              f"{out['dprime_stored_mean']:.4f} standalone {out['dprime_standalone_mean']:.4f} | "
              f"{'PASS' if out['pass'] else 'FAIL'}", flush=True)
    res.update(pass_all=bool(ok), tol_emb_rel=TOL_EMB_REL, tol_dprime=TOL_DPRIME, dbspl=DBSPL, device=str(device),
               host=socket.gethostname().split('.')[0], repo_sha=repo_sha, analysis_code_sha=analysis_sha,
               date=time.strftime('%F %T'))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(f'[save] {OUT_JSON} ({(time.time() - t0) / 60:.1f} min)')
    print('ESC-50 D-PRIME ' + ('PASSED' if ok else 'FAILED'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
