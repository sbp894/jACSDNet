"""Convert J0 (joint-AC/SDNet) seed 0 into the standalone jACSDNet v0 checkpoint.

LBHB-internal: imports the training code (neural_auditory_categorization/ACnetTCnet_joint) and reads the
original checkpoint. Run once, under a torch env, through the launch-ssh-job skill:

    python data/convert_j0_to_standalone.py

Source (the blessed artifact; nothing is retrained or chosen here):
    ACnetTCnet_joint/output/checkpoints/<TAG>.pt              full model (trunk + 62 site heads + classifier)
    ACnetTCnet_joint/output/checkpoints/<TAG>_acnetview.pt    trunk + site heads as MT_ResNet_v2, with the
                                                              front end the ESC-50 readouts used
TAG = jmc_p62_mega4k12_clfcnn10p128_eps0.3_L0.5_seed0_asb256_bnb256_asg2.985_spe1360_tb30_cratchet_refetch

What it does, on CPU (determinism):
  1. Rebuild the original model (train_joint_megaclf.load_joint_megaclf) and the acnetview front end exactly as
     jmc_readouts.py does (60 dB SPL, level_mode 'exact', lbhb_mode False, no as_gain).
  2. Copy weights into jacsdnet_model.JACSDNet: trunk as is; the 62 site heads concatenated into one Linear +
     DEXP (all 62 must be dexp ON); the projection stem, CNN10 and logits as is.
  3. Parity: on the example clip and 4 ESC-50 500-ms clips, every readout (mf, mr, proj, clf, logits) of the
     standalone must equal the original's (jmc_readouts.window_features for clf / logits), and the standalone
     front end must equal the acnetview's. Aborts before saving on any mismatch.
  4. Write weights/jacsdnet_v0.pt and validation/parity_fixture.npz (the original's outputs, for
     validation/check_parity.py, which runs anywhere).
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOINT = '/auto/users/satya/code/projects_getting_started/neural_auditory_categorization/ACnetTCnet_joint/'
TCMEGA = '/auto/users/satya/code/projects_getting_started/neural_auditory_categorization/TCnet_mega/'
TAG = 'jmc_p62_mega4k12_clfcnn10p128_eps0.3_L0.5_seed0_asb256_bnb256_asg2.985_spe1360_tb30_cratchet_refetch'
CKPT = JOINT + 'output/checkpoints/' + TAG + '.pt'
VIEW = JOINT + 'output/checkpoints/' + TAG + '_acnetview.pt'
EXAMPLE_WAV = os.path.join(REPO, 'examples', 'example_esc50_clip.wav')
ESC50_DIR = '/auto/users/satya/code/projects_getting_started/downloaded/dataset/ESC-50-master/misc_out/ESC50_500ms_v0/'
N_ESC50_FIXTURE = 4
OUT_PT = os.path.join(REPO, 'weights', 'jacsdnet_v0.pt')
OUT_FIXTURE = os.path.join(REPO, 'validation', 'parity_fixture.npz')
TOL_REL = 1e-5            # CPU float32, same ops in the same order: expect ~0

for _p in (JOINT, TCMEGA, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torchaudio                                          # noqa: E402
import train_joint_megaclf as jmc                          # noqa: E402
import jmc_readouts as jr                                  # noqa: E402
import esc50_mr_embeddings as eme                          # noqa: E402
from tcnet_mega_dataset import load_strong_vocab           # noqa: E402
import jacsdnet_model as jm_sa                             # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def git_sha(path):
    try:
        sha = subprocess.check_output(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = subprocess.run(['git', '-C', path, 'diff', '--quiet', 'HEAD', '--'], capture_output=True).returncode
        return sha + ('-dirty' if dirty else '')
    except Exception as ex:                                # a fresh repo with no commit yet
        return f'<unavailable: {type(ex).__name__}>'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel_max(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def main():
    device = torch.device('cpu')
    print(f'[src] {CKPT}\n[src] {VIEW}')

    # ---- 1. the original model and front end, exactly as jmc_readouts.main builds them
    view, model_params, data_params, n_heads, hod, rtest_view = eme._build_joint(dict(ckpt=VIEW, note=''), device)
    view.params_audio_proc['model_mode'] = 'audio'
    view.update_audio_process(params_audio_proc={'overall_db': eme.DBSPL, 'level_mode': eme.LEVEL_MODE,
                                                 'lbhb_mode': eme.LBHB_MODE}, verbose=True)
    view.audio_process.device = device                     # update_audio_process rebuilds it on the GPU if one exists
    view.eval()
    joint, blob = jmc.load_joint_megaclf(CKPT, device)
    n_same = jr.assert_same_trunk(joint, view)
    print(f'[src] full ckpt trunk + site heads == acnetview ({n_same} tensors)')
    assert blob['clf_ctor_args'] == dict(arch='cnn10', d_in=3124, n_classes=456, window_bins=50, proj=128), \
        blob['clf_ctor_args']
    nl = list(blob['head_nonlinearity'])
    assert len(nl) == 62 and all(m == 'dexp' for m in nl), f'site-head nonlinearities: {sorted(set(nl))}'
    n_sites = len(blob['site_head_dims'])
    dims = [int(d) for d in blob['site_head_dims']]
    assert sum(dims) == 3124, sum(dims)

    # ---- 2. the standalone model
    cfg = dict(jm_sa.DEFAULT_CONFIG)
    tk = blob['trunk_kwargs']
    cfg.update({'hidden_dim': list(tk['hidden_dim']), 'kernel_size': list(tk['kernel_size']),
                'n_neurons': int(sum(dims)), 'norm_1d': bool(tk['norm_1d']), 'skip': bool(tk['skip']),
                'res_scale': float(tk['res_scale']), 'overall_db': float(eme.DBSPL),
                'compress': data_params['compress'], 'level_mode': eme.LEVEL_MODE, 'lbhb_mode': eme.LBHB_MODE})
    assert cfg['compress'] == 'sqrt' and list(tk['time_pool']) == [1, 1, 1, 1], (cfg['compress'], tk['time_pool'])
    sa = jm_sa.JACSDNet(cfg).to(device).eval()

    sa.layers_shared.load_state_dict(joint.layers_shared.state_dict(), strict=True)
    lin_w, lin_b, base, amp, kappa = [], [], [], [], []
    for k in range(n_sites):
        linear, dexp = joint.site_heads[k][0], joint.site_heads[k][1]
        assert linear.weight.shape[0] == dims[k], (k, linear.weight.shape, dims[k])
        lin_w.append(linear.weight.data); lin_b.append(linear.bias.data)
        base.append(dexp.base.data); amp.append(dexp.amp.data); kappa.append(dexp.kappa.data)
    with torch.no_grad():
        sa.readout_linear.weight.copy_(torch.cat(lin_w, 0)); sa.readout_linear.bias.copy_(torch.cat(lin_b, 0))
        sa.readout_nl.base.copy_(torch.cat(base, 0)); sa.readout_nl.amp.copy_(torch.cat(amp, 0))
        sa.readout_nl.kappa.copy_(torch.cat(kappa, 0))
    head = joint.mega.head
    sa.proj_stem.load_state_dict(head.stem.state_dict(), strict=True)
    sa.cnn10.load_state_dict(head.backbone.state_dict(), strict=True)
    sa.logits.load_state_dict(head.logits.state_dict(), strict=True)
    n_sa = sum(p.numel() for p in sa.parameters())
    print(f'[sa] standalone built: {n_sa:,} parameters')

    # ---- 3. parity
    esc_files = sorted(f for f in os.listdir(ESC50_DIR) if f.endswith('.wav'))[:N_ESC50_FIXTURE]
    items = [('example_esc50_clip.wav', EXAMPLE_WAV)] + [(f, ESC50_DIR + f) for f in esc_files]
    fixture, report = {}, {}
    with torch.no_grad():
        for i, (name, path) in enumerate(items):
            w, fs = torchaudio.load(path)
            gtg_ref = view.audio_process(w, fs).squeeze().cpu()              # (T, 32), the original front end
            gtg_sa = sa._to_gtg(path)                                         # stdlib wav loader + copied front end
            r_gtg = rel_max(gtg_sa.numpy(), gtg_ref.numpy())
            x = gtg_ref[None]
            h_ref = joint.layers_shared(x)
            mr_ref = joint.rates(h_ref)
            ref = {'mf': h_ref, 'mr': mr_ref, 'proj': head.stem(mr_ref, None)}
            wf = jr.window_features(joint, x)                                 # the function the stored CLF dumps used
            ref['clf'] = torch.from_numpy(wf['clf_flatten'])
            ref['logits'] = torch.from_numpy(wf['clf_logits'])
            row = {'gtg': r_gtg}
            for ro in jm_sa.READOUTS:
                out = sa.readout_from_model_input(x, ro)
                row[ro] = rel_max(out.numpy(), ref[ro].numpy())
                fixture[f'{i}_{ro}'] = ref[ro].numpy().astype(np.float32)
            fixture[f'{i}_gtg'] = gtg_ref.numpy().astype(np.float32)
            report[name] = row
            print(f'[parity] {name:32s} T={gtg_ref.shape[0]:4d} ' + ' '.join(f'{k}={v:.1e}' for k, v in row.items()))
    worst = max(v for row in report.values() for v in row.values())
    assert worst < TOL_REL, f'PARITY FAILED: worst rel max|d| {worst:.2e} >= {TOL_REL}'
    print(f'[parity] PASSED: worst rel max|d| {worst:.2e} over {len(items)} inputs x (gtg + {len(jm_sa.READOUTS)} readouts)')

    # ---- 4. save
    mids, _, labels = load_strong_vocab()
    assert len(labels) == 456, len(labels)
    cell_names = [str(c) for site in blob['cell_names'] for c in np.atleast_1d(site)]
    assert len(cell_names) == 3124, len(cell_names)
    m = blob['metrics']
    cell_rtest = np.asarray(m['rtest_percell'], dtype=np.float32)
    provenance = dict(
        source_model='J0 (joint-AC/SDNet), seed 0', source_checkpoint=CKPT, source_checkpoint_sha256=sha256(CKPT),
        source_acnetview=VIEW, source_train_git_sha=blob['git_sha'], source_train_host=blob['host'],
        source_tag=blob['tag'], source_eps=float(blob['eps']), source_lam=float(blob['lam']),
        source_as_gain=float(blob['as_gain']), source_stop_reason=blob.get('stop_reason'),
        source_best_epochs=blob.get('best_epochs'), source_bnrecal=blob.get('bnrecal'),
        rtest_cell_mean=float(m['cell_mean_rtest']), rtest_site_mean=float(m['site_mean_rtest']),
        audioset_test_mAP=float(m['as_test']['mAP']), audioset_val_mAP=float(m['as_val']['mAP']),
        site_head_dims=dims, site_index=[int(s) for s in blob['site_index']],
        frontend_note=('ESC-50 / downstream readouts: 60 dB SPL, level_mode exact, lbhb_mode False, sqrt, no as_gain. '
                       f"AudioSet training inputs were scaled x{blob['as_gain']} in the sqrt domain "
                       '(= +19 dB waveform level); the default front end does not apply it.'),
        parity=report, parity_tol_rel=TOL_REL,
        converted_by='data/convert_j0_to_standalone.py', converted_date=datetime.now(timezone.utc).isoformat(),
        converter_repo_sha=git_sha(REPO), training_code_repo_sha=git_sha(JOINT),
    )
    cfg['version'] = jm_sa.JACSDNET_VERSION
    torch.save({'version': jm_sa.JACSDNET_VERSION, 'model_name': jm_sa.MODEL_NAME,
                'state_dict': sa.state_dict(), 'config': cfg, 'cell_names': cell_names,
                'cell_rtest': torch.from_numpy(cell_rtest),          # a tensor, so weights_only=True loads it
                'audioset_labels': list(labels), 'audioset_mids': list(mids),
                'provenance': provenance}, OUT_PT)
    print(f'[save] {OUT_PT} ({os.path.getsize(OUT_PT) / 1e6:.1f} MB)')
    np.savez_compressed(OUT_FIXTURE, names=np.array([n for n, _ in items]), readouts=np.array(jm_sa.READOUTS),
                        provenance=np.array(json.dumps(dict(source_checkpoint=CKPT, converter_repo_sha=provenance['converter_repo_sha'],
                                                            made_by='data/convert_j0_to_standalone.py',
                                                            made=provenance['converted_date']))),
                        **fixture)
    print(f'[save] {OUT_FIXTURE} ({os.path.getsize(OUT_FIXTURE) / 1e6:.1f} MB)')

    # ---- reload round trip through the public loader
    sa2, rt = jm_sa.load_jacsdnet(OUT_PT, map_location='cpu')
    with torch.no_grad():
        for ro in jm_sa.READOUTS:
            x = torch.from_numpy(fixture['0_gtg'])
            d = rel_max(sa2.readout_from_model_input(x, ro).numpy(), fixture[f'0_{ro}'])
            assert d < TOL_REL, f'reload {ro}: {d:.2e}'
    assert np.allclose(rt, cell_rtest) and len(sa2.audioset_labels) == 456
    print('[reload] load_jacsdnet round trip matches the fixture on every readout')
    print('DONE')


if __name__ == '__main__':
    main()
