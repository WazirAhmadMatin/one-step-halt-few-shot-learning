"""
smoke_test.py (v4) - quick check (about 10-15 min, synthetic data, no images, no backbone download) that the v4 notebook and run_all.py work on
this machine, for all four dataset x backbone combinations, including the parallel mode.

Put it in the same folder as run_all.py and onestep_halt_revision_v4.ipynb, then in Anaconda Prompt:
    python smoke_test.py --shards 5
Your real results are never touched. The accuracy numbers it prints are meaningless; only PASS / FAIL matters.
"""
import argparse, os, subprocess, sys, time, shutil, json

HERE = os.path.dirname(os.path.abspath(__file__))
COMBOS = ['ecu_clip', 'ecu_dinov3', 'deepseagrass_clip', 'deepseagrass_dinov3']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--shards', type=int, default=1); a = ap.parse_args()
    for f in ('run_all.py', 'onestep_halt_revision_v4.ipynb'):
        if not os.path.exists(os.path.join(HERE, f)): sys.exit(f'FAIL: {f} must be in the same folder as smoke_test.py ({HERE})')
    missing = []
    for m in ('numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'matplotlib', 'PIL'):
        try: __import__(m)
        except ImportError: missing.append(m)
    if missing: sys.exit(f'FAIL: missing Python packages {missing} - activate the environment you use for the notebook')
    import torch
    try: import transformers; tv = transformers.__version__
    except ImportError: tv = 'NOT INSTALLED (needed for the real run: pip install -U transformers)'
    print(f'python {sys.version.split()[0]} | torch {torch.__version__} | transformers {tv} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"}')
    smoke_dir = os.path.join(HERE, 'smoke_test_v4'); shutil.rmtree(smoke_dir, ignore_errors=True); t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, 'run_all.py'), '--stages', 'dev,final', '--all-smoke', '--shards', str(a.shards), '--smoke-dir', smoke_dir], cwd=HERE)
    if r.returncode != 0: sys.exit('\nSMOKE TEST FAILED - send Claude the log file named in the ERROR line above')
    root = lambda c: smoke_dir if c == 'ecu_clip' else os.path.join(smoke_dir, 'runs', c)
    need = [os.path.join(root(c), 'dev', 'v4', 'frozen_config.json') for c in COMBOS if c.startswith('ecu')] + \
           [os.path.join(root(c), 'final', 'v4', f) for c in COMBOS for f in ('latex_tables.tex', 'paired_tests_acc.csv', 'paired_tests_macro_f1.csv', 'summary_all_methods.csv',
                                                                         'leakage_diagnostic.json', 'teacher_calibration.csv', 'fig_confusion.png', 'fig_selection_map.png')] + \
           [os.path.join(smoke_dir, 'summary_v4', f) for f in ('summary_all_combinations.csv', 'summary_all_combinations.tex', 'fig_all_combinations.png')]
    miss = [p for p in need if not os.path.exists(p)]
    if miss: sys.exit(f'SMOKE TEST FAILED: output files missing {miss}')
    fz = json.load(open(os.path.join(root('ecu_dinov3'), 'dev', 'v4', 'frozen_config.json'))); fc = json.load(open(os.path.join(root('ecu_clip'), 'dev', 'v4', 'frozen_config.json')))
    if fz['NEW_METHOD'] != fc['NEW_METHOD']: sys.exit('SMOKE TEST FAILED: the DINOv3 run did not keep the CLIP method')
    print(f'\nSMOKE TEST PASSED for all four combinations in {(time.time() - t0) / 60:.1f} min ({a.shards} process(es)).')
    import pandas as pd; st = pd.read_csv(os.path.join(smoke_dir, 'summary_v4', 'summary_all_combinations.csv'))
    if sorted(set(zip(st.dataset, st.backbone))) != sorted({('DeepSeagrass', 'CLIP ViT-B/32'), ('DeepSeagrass', 'DINOv3 ViT-B/16'), ('ECU-MSS', 'CLIP ViT-B/32'), ('ECU-MSS', 'DINOv3 ViT-B/16')}):
        sys.exit('SMOKE TEST FAILED: the cross-combination summary does not contain all four combinations')
    print(f'Next: python run_all.py --shards {max(a.shards, 1)} --stages dev,final')
    shutil.rmtree(smoke_dir, ignore_errors=True)

if __name__ == '__main__':
    main()
