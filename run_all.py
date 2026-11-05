"""
run_all.py  (v4)  -  runs the whole One-Step HALT revision pipeline in one go, for 2 datasets x 2 frozen backbones:

    ecu_clip            ECU-MSS + CLIP ViT-B/32    dev (tuning on the photo-level validation split, method decision) + final (test)
    ecu_dinov3          ECU-MSS + DINOv3 ViT-B/16  dev (knobs re-tuned for this backbone; method = the one chosen with CLIP) + final
    deepseagrass_clip   DeepSeagrass + CLIP        final only: EVERYTHING frozen from ecu_clip (no tuning on DeepSeagrass)
    deepseagrass_dinov3 DeepSeagrass + DINOv3      final only: everything frozen from ecu_dinov3

Order: smoke test of all four (synthetic data) -> ecu_clip -> ecu_dinov3 -> deepseagrass_clip -> deepseagrass_dinov3, with checks in between.

Usage (Anaconda Prompt, in the folder with this file and onestep_halt_revision_v4.ipynb):
    python run_all.py --shards 5                                  # everything (about 20-30 h)
    python run_all.py --shards 5 --stages dev,final               # everything except the smoke test
    python run_all.py --shards 5 --combos ecu_dinov3,deepseagrass_dinov3   # only some combinations
    python run_all.py --stages smoke --shards 5                   # only the smoke test (~10 min)

Resumable: start the same command again after any interruption; finished runs are skipped. Feature caches of earlier versions are reused.
Logs: <this folder>/logs/<combo>_<stage>_shard<i>.log
Results: C:/journal paper/clip/{dev,final}/v4 (ecu_clip), C:/journal paper/clip/runs/<combo>/{dev,final}/v4 (the others), C:/journal paper/clip/summary_v4 (all four)
"""
import argparse, json, os, shutil, subprocess, sys, time, io

HERE = os.path.dirname(os.path.abspath(__file__))
ALL_COMBOS = ['ecu_clip', 'ecu_dinov3', 'deepseagrass_clip', 'deepseagrass_dinov3']


def notebook_code(nb_path, drop_last_cell=False):
    nb = json.load(open(nb_path, encoding='utf-8'))
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    head = cells[0]
    for need in ("os.environ.get('HALT_STAGE'", "os.environ.get('HALT_SMOKE'", "os.environ.get('HALT_SHARD_ID'", "os.environ.get('HALT_WAIT'",
                 "os.environ.get('HALT_DATASET'", "os.environ.get('HALT_BACKBONE'"):
        if need not in head:
            sys.exit(f'ERROR: the first code cell of {os.path.basename(nb_path)} does not read {need}...).\n'
                     f'Use the unmodified v4 notebook (do not hard-code STAGE / SMOKE_TEST / SHARD_ID / DATASET / BACKBONE in it when using run_all.py).')
    if drop_last_cell: cells = [c for c in cells if not c.lstrip().startswith('# MAIN_SHARD_ONLY')]   # tables, figures, summary: shard 0 only
    return cells


class Tee(io.TextIOBase):
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            try: st.write(s); st.flush()
            except UnicodeEncodeError: st.write(s.encode('ascii', 'replace').decode()); st.flush()
        return len(s)
    def flush(self):
        for st in self.streams: st.flush()


def worker(args):
    """executes the notebook code cells in this process (no Jupyter kernel needed)"""
    log = open(args.log, 'a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = Tee(log, sys.__stdout__) if args.shard == 0 else Tee(log)
    print(f'\n######## {time.strftime("%Y-%m-%d %H:%M:%S")}  {os.environ["HALT_DATASET"]} + {os.environ["HALT_BACKBONE"]}  stage={os.environ["HALT_STAGE"]}  '
          f'smoke={os.environ["HALT_SMOKE"]}  shard {args.shard}/{args.nshards}')
    g = {'__name__': '__halt_notebook__'}
    for i, src in enumerate(notebook_code(args.notebook, drop_last_cell=args.shard != 0)):
        print(f'\n======== code cell {i + 1}'); t0 = time.time()
        exec(compile(src, f'<notebook code cell {i + 1}>', 'exec'), g)
        print(f'-------- code cell {i + 1} done in {(time.time() - t0) / 60:.1f} min')
    print(f'######## stage finished OK (shard {args.shard})')


def run_stage(combo, stage, smoke, nshards, nb, smoke_dir):
    ds, bb = combo.split('_')
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    tag = ('smoke_' if smoke else '') + f'{combo}_{stage}'
    env = dict(os.environ, HALT_STAGE=stage, HALT_SMOKE='1' if smoke else '0', HALT_SMOKE_DIR=smoke_dir, HALT_N_SHARDS=str(nshards), HALT_DATASET=ds, HALT_BACKBONE=bb,
               HALT_WAIT='1', MPLBACKEND='Agg', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    procs = []
    print(f'\n=== {tag}: starting {nshards} process(es)   (shard 0 prints here; all shards log to logs\\{tag}_shard<i>.log)')
    t0 = time.time()
    for i in range(nshards):
        log = os.path.join(HERE, 'logs', f'{tag}_shard{i}.log')
        procs.append(subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--worker', '--shard', str(i), '--nshards', str(nshards),
                                       '--notebook', nb, '--log', log], env=dict(env, HALT_SHARD_ID=str(i)), cwd=HERE))
        if i == 0 and nshards > 1: time.sleep(20)          # let shard 0 start first (it builds caches and helper files)
    try:
        while True:
            codes = [p.poll() for p in procs]
            bad = [i for i, c in enumerate(codes) if c not in (None, 0)]
            if bad:
                for p in procs:
                    if p.poll() is None: p.kill()
                sys.exit(f'\nERROR in {tag}: shard(s) {bad} failed - see logs\\{tag}_shard{bad[0]}.log (the end of the file shows the error). '
                         f'Fix it and start the same command again: finished runs are kept.')
            if all(c == 0 for c in codes): break
            time.sleep(5)
    except KeyboardInterrupt:
        for p in procs: p.kill()
        sys.exit('\nstopped by Ctrl+C - start the same command again to continue (finished runs are kept)')
    print(f'=== {tag}: finished in {(time.time() - t0) / 60:.1f} min')


def project_dir(nb):
    """read PROJECT from the notebook's first code cell (the real-data folder)"""
    for line in notebook_code(nb)[0].splitlines():
        if line.strip().startswith('PROJECT') and "r'" in line:
            return line.split("r'")[1].split("'")[0]
    sys.exit('could not find PROJECT in the first code cell')


def combo_root(base, combo):
    return base if combo == 'ecu_clip' else os.path.join(base, 'runs', combo)       # same rule as the notebook


def check_after_dev(base, combo):
    d = os.path.join(combo_root(base, combo), 'dev', 'v4')
    fz_path = os.path.join(d, 'frozen_config.json'); lk_path = os.path.join(d, 'leakage_diagnostic.json')
    if not os.path.exists(fz_path): sys.exit(f'ERROR: {fz_path} not found - the {combo} dev stage did not finish')
    fz = json.load(open(fz_path))
    if not fz.get('NEW_METHOD'): sys.exit(f'ERROR: {fz_path} has no NEW_METHOD - the {combo} dev stage did not reach the decision cell')
    lk = json.load(open(lk_path)) if os.path.exists(lk_path) else {}
    print(f'\n=== {combo}: dev stage checks')
    if 'photo_split' in lk:
        for s in ('val', 'test'): print(f"  {s}: nearest train patch from the same photograph: {100 * lk['photo_split'][s]['frac_nn_same_photo']:.2f}%  (must be ~0)")
        print('  validation split:', lk.get('val_split_info'))
    print(f"  NEW_METHOD = {fz['NEW_METHOD']} (rule's own choice on this backbone: {fz.get('own_rule_choice')}) | any eligible: {fz.get('any_eligible')} | "
          f"beats random on dev: {fz.get('new_beats_random_on_dev')} | beats baseline on dev: {fz.get('new_beats_baseline_on_dev')}")
    print(f"  frozen: tau={fz['TAU']} w={fz['W_CONF']} tau_gate={fz['TAU_GATE']} w_norm={fz['W_NORM']} semi={fz['SEMI_CFG']}")


def report_leak(base, combo):
    lk_path = os.path.join(combo_root(base, combo), 'final', 'v4', 'leakage_diagnostic.json')
    if not os.path.exists(lk_path): return
    lk = json.load(open(lk_path)); p = lk.get('photo_nn_profile', {})
    if p:
        print(f"  {combo} split check: base rate {100 * p['base_rate']:.1f}% | test as close as a same-photo patch: {100 * p['test']['frac_as_close_as_same_photo']:.1f}%"
              + ('  <-- WARNING, tell Claude' if p['test'].get('warning') else '  (OK)'))
    if 'exact_duplicates' in lk: print(f"  {combo} exact duplicate files also in training: {lk['exact_duplicates']}")


def run_plan(combos, stages, smoke, nshards, nb, base, smoke_dir):
    for combo in combos:
        ds, bb = combo.split('_')
        if ds == 'ecu':
            if bb != 'clip' and not os.path.exists(os.path.join(combo_root(base, 'ecu_clip'), 'dev', 'v4', 'frozen_config.json')):
                sys.exit(f'ERROR: {combo} needs the ECU-MSS + CLIP dev stage first (its method choice). Include ecu_clip in --combos.')
            if 'dev' in stages: run_stage(combo, 'dev', smoke, nshards, nb, smoke_dir); check_after_dev(base, combo)
            if 'final' in stages:
                if 'dev' not in stages: check_after_dev(base, combo)
                run_stage(combo, 'final', smoke, nshards, nb, smoke_dir)
        else:
            src = f'ecu_{bb}'
            if 'final' in stages:
                if not os.path.exists(os.path.join(combo_root(base, src), 'dev', 'v4', 'frozen_config.json')):
                    sys.exit(f'ERROR: {combo} uses the settings frozen on {src}: run {src} (dev) first')
                run_stage(combo, 'final', smoke, nshards, nb, smoke_dir); report_leak(base, combo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', default='smoke,dev,final', help='comma list from: smoke, dev, final')
    ap.add_argument('--combos', default=','.join(ALL_COMBOS), help='comma list from: ' + ', '.join(ALL_COMBOS))
    ap.add_argument('--shards', type=int, default=1, help='parallel processes (try 4-6 on the RTX PRO 4000; 1 = no parallelism)')
    ap.add_argument('--notebook', default=os.path.join(HERE, 'onestep_halt_revision_v4.ipynb'))
    ap.add_argument('--smoke-dir', default=os.path.join(HERE, 'smoke_run_v4'))
    ap.add_argument('--all-smoke', action='store_true', help=argparse.SUPPRESS)    # testing: run the real plan on synthetic data
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--shard', type=int, default=0, help=argparse.SUPPRESS); ap.add_argument('--nshards', type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument('--log', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker: return worker(args)

    nb = os.path.abspath(args.notebook)
    if not os.path.exists(nb): sys.exit(f'notebook not found: {nb}')
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    combos = [c.strip() for c in args.combos.split(',') if c.strip()]
    for s in stages:
        if s not in ('smoke', 'dev', 'final'): sys.exit(f'unknown stage {s}')
    for c in combos:
        if c not in ALL_COMBOS: sys.exit(f'unknown combination {c} (choose from {ALL_COMBOS})')
    combos = [c for c in ALL_COMBOS if c in combos]                    # always in dependency order
    proj = project_dir(nb); base = args.smoke_dir if args.all_smoke else proj
    print(f'notebook: {nb}\nresults : {base}\nstages  : {stages} | combinations: {combos} | shards: {args.shards}')
    T0 = time.time()
    if 'smoke' in stages:
        shutil.rmtree(args.smoke_dir, ignore_errors=True)
        run_plan(ALL_COMBOS, ['dev', 'final'], True, args.shards, nb, args.smoke_dir, args.smoke_dir)
        print('\n=== smoke test passed for all four combinations (synthetic data; the numbers mean nothing)')
        shutil.rmtree(args.smoke_dir, ignore_errors=True)
    real = [s for s in stages if s != 'smoke']
    if real:
        run_plan(combos, real, args.all_smoke, args.shards, nb, base, args.smoke_dir)
        if 'final' in real:
            print('\n=== FINAL stages done. Send these folders to Claude (latex_tables.tex, paired_tests_acc.csv, summary_all_methods.csv, selection_diagnostics.csv, '
                  'timing_per_stage.csv, leakage_diagnostic.json) and the logs\\*_shard0.log files:')
            for c in combos: print('   ', os.path.join(combo_root(base, c), 'final', 'v4'))
            print('    and the cross-combination summary:', os.path.join(base, 'summary_v4'))
    print(f'\nall requested stages finished in {(time.time() - T0) / 3600:.2f} h')


if __name__ == '__main__':
    main()
