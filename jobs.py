"""
Job list for the TMLR experiments and a runner that splits it across the Kaggle accounts.

Every job is (phase, run, dataset, seed) and writes to <out>/<run>/seed<seed>/<dataset>/summary.json,
the same layout as the pilot. Each phase is split over the accounts by estimated GPU time
(largest jobs first, each to the least loaded account), so the split is deterministic and every
account computes the same assignment. To move jobs to another account, edit OVERRIDES.

  python jobs.py list                                # every job, its account and estimated hours
  python jobs.py list --friend 2 --phases phase1     # one account's jobs
  python jobs.py run --friend 2 --phases phase1 base15 --gpus 0 1 --budget_h 11.3

`run` keeps one job per GPU, skips jobs that already have a summary.json, does not start a job
that would not finish inside the session budget, and zips <out> after every finished job.
"""
import os
import sys
import json
import time
import shutil
import threading
import subprocess
from argparse import ArgumentParser

N_FRIENDS = 4
SEEDS = [0, 1, 2]
PILOT = ['cifar', 'dtd', 'dsprites_loc', 'smallnorb_azi']
STRUCT6 = ['clevr_count', 'clevr_dist', 'dmlab', 'kitti', 'dsprites_ori', 'smallnorb_ele']
NEW = ['caltech101', 'oxford_flowers102', 'oxford_iiit_pet', 'svhn', 'sun397', 'patch_camelyon', 'eurosat',
       'resisc45', 'diabetic_retinopathy', 'clevr_count', 'clevr_dist', 'dmlab', 'kitti', 'dsprites_ori',
       'smallnorb_ele']
N_TEST = {'cifar': 10000, 'caltech101': 6084, 'dtd': 1880, 'oxford_flowers102': 6149, 'oxford_iiit_pet': 3669,
          'svhn': 26032, 'sun397': 21750, 'patch_camelyon': 32768, 'eurosat': 5400, 'resisc45': 6300,
          'diabetic_retinopathy': 42670, 'clevr_count': 15000, 'clevr_dist': 15000, 'dmlab': 22735,
          'kitti': 711, 'dsprites_loc': 73728, 'dsprites_ori': 73728, 'smallnorb_azi': 12150,
          'smallnorb_ele': 12150}
COMMON = '--method adaptformer --dim 8 --bit 32 --load_config'
# large test sets: evaluate at epochs 24/49/74/99 and skip the full-branch eval (as dsprites_loc in the pilot)
BIG_TEST = 20000
TEACHER = '{teachers}/adaptformer/seed{seed}/{dataset}.pth'

# run name -> (main.py flags, epochs). kd-* runs need the pilot AdaptFormer checkpoint of the same seed.
RUNS = {
    'adaptformer': ('', 100),
    'fpet-r2': ('--r_layer 2', 100),
    'fpet-r4': ('--r_layer 4', 100),
    'mcpet-r2': ('--r_layer 2 --consistency', 100),
    'mcpet-r4': ('--r_layer 4 --consistency', 100),
    # phase 1: go/no-go
    'kd-r2': ('--r_layer 2 --teacher_ckpt ' + TEACHER, 100),
    'kd-r4': ('--r_layer 4 --teacher_ckpt ' + TEACHER, 100),
    # compute-matched FPET (~1.6x the epochs); --eval_every 16 keeps 10 evaluations like the 100-epoch runs
    'fpet-r2-e160': ('--r_layer 2 --epochs 160 --eval_every 16', 160),
    'fpet-r4-e160': ('--r_layer 4 --epochs 160 --eval_every 16', 160),
    # loss ablation (M1). lambda_f = 0 runs the full path without gradients: the gradients are identical
    # (the KL target is detached), and the run doubles as the no-grad-teacher memory measurement (M7)
    'mcpet-r2-nokl': ('--r_layer 2 --consistency --cons_weight 0', 100),
    'mcpet-r2-nofullce': ('--r_layer 2 --consistency --full_ce_weight 0 --full_no_grad', 100),
    'mcpet-r2-feat': ('--r_layer 2 --consistency --feat_weight 1', 100),
    # week 3: cheaper training
    'mcpet-r2-every2': ('--r_layer 2 --consistency --full_every 2', 100),
    'mcpet-r2-every4': ('--r_layer 2 --consistency --full_every 4', 100),
    # ---- revision (reviewer M1-M3, M5, M7, minor) ----
    # M1.3 stochastic merging: one forward per step, merged with p = 0.5 (FPET's memory)
    'fpet-r2-stoch': ('--r_layer 2 --merge_prob 0.5', 100),
    # M2 tied drop-path masks; the RNG stream then matches FPET exactly (tighter pairing)
    'mcpet-r2-tied': ('--r_layer 2 --consistency --tie_masks', 100),
    # M2 no stochastic regularisation at all (drop path and adapter dropout off), both methods
    'fpet-r2-nodrop': ('--r_layer 2 --drop_path 0 --adapter_dropout 0', 100),
    'mcpet-r2-nodrop': ('--r_layer 2 --consistency --drop_path 0 --adapter_dropout 0', 100),
    # M3 checkpoint selection on val200 (train on train800 only)
    'fpet-r2-val': ('--r_layer 2 --val_select', 100),
    'mcpet-r2-val': ('--r_layer 2 --consistency --val_select', 100),
    # minor: lambda_kl and temperature sweep
    'mcpet-r2-kl0.5': ('--r_layer 2 --consistency --cons_weight 0.5', 100),
    'mcpet-r2-kl2': ('--r_layer 2 --consistency --cons_weight 2', 100),
    'mcpet-r2-kl4': ('--r_layer 2 --consistency --cons_weight 4', 100),
    'mcpet-r2-T2': ('--r_layer 2 --consistency --cons_temp 2', 100),
    # M5 second PET method (LoRA); method flag is replaced in command()
    'lora': ('--method lora', 100),
    'lora-fpet-r2': ('--method lora --r_layer 2', 100),
    'lora-mcpet-r2': ('--method lora --r_layer 2 --consistency', 100),
}
for L in [1, 3, 6, 8]:
    RUNS[f'fpet-r{L}'] = (f'--r_layer {L}', 100)
    RUNS[f'mcpet-r{L}'] = (f'--r_layer {L} --consistency', 100)

PHASES = {
    # kd-* teachers whose pilot checkpoint is not on disk (only seed 0 of cifar/dtd/smallnorb_azi is);
    # skip this phase if the .pth files are recovered from the pilot's Kaggle outputs
    'teachers': [('adaptformer', d, s) for d in PILOT for s in SEEDS
                 if not (s == 0 and d != 'dsprites_loc')],
    # week 1: is MC-FPET better than plain distillation / longer training? which loss terms matter?
    'phase1': [(r, d, s) for r in ['kd-r2', 'kd-r4', 'fpet-r2-e160', 'fpet-r4-e160',
                                   'mcpet-r2-nokl', 'mcpet-r2-nofullce', 'mcpet-r2-feat']
               for d in PILOT for s in SEEDS],
    # weeks 1-2: baselines on the 15 new tasks (needed whatever phase 1 shows)
    'base15': [(r, d, s) for r in ['adaptformer', 'fpet-r2', 'fpet-r4'] for d in NEW for s in SEEDS],
    # week 2 (after a go): main result
    'mc15-r2': [('mcpet-r2', d, s) for d in NEW for s in SEEDS],
    # week 3
    'mc15-r4': [('mcpet-r4', d, s) for d in NEW for s in SEEDS],
    'memfix': [(r, d, s) for r in ['mcpet-r2-every2', 'mcpet-r2-every4'] for d in PILOT for s in SEEDS],
    'sweep': [(f'{m}-r{L}', d, 0) for L in [1, 3, 6, 8] for m in ['fpet', 'mcpet'] for d in PILOT],
    # ---------------- revision, in the reviewer's priority order ----------------
    # R1 (M1, M7): loss ablation, stochastic merging, compute-matched FPET
    'rev-ablate': [(r, d, s) for r in ['mcpet-r2-nokl', 'mcpet-r2-nofullce', 'fpet-r2-stoch', 'fpet-r2-e160']
                   for d in PILOT for s in SEEDS],
    # R2 (M2): tied masks on all tasks; drop path / dropout off on sNORB-Azim
    'rev-dropout': [('mcpet-r2-tied', d, s) for d in PILOT for s in SEEDS] +
                   [(r, 'smallnorb_azi', s) for r in ['fpet-r2-nodrop', 'mcpet-r2-nodrop'] for s in SEEDS],
    # R3 (M6): external frozen AdaptFormer teacher; needs the 'teachers' phase or recovered checkpoints
    'rev-kd': [('kd-r2', d, s) for d in PILOT for s in SEEDS],
    # R4 (M3): validation-based checkpoint selection, one sensitivity seed
    'rev-val': [(r, d, 0) for r in ['fpet-r2-val', 'mcpet-r2-val'] for d in PILOT],
    # R5 (M5): the six remaining Structured tasks at layer 2
    'rev-struct': [(r, d, s) for r in ['adaptformer', 'fpet-r2', 'mcpet-r2'] for d in STRUCT6 for s in SEEDS],
    # R6 (M7): one cheaper-training data point (every 2nd step); no-grad teacher is mcpet-r2-nofullce
    'rev-cost': [('mcpet-r2-every2', d, s) for d in PILOT for s in SEEDS],
    # R7 (minor): lambda_kl / temperature sweep on CIFAR-100, where MC-FPET regressed
    'rev-sweep': [(r, 'cifar', s) for r in ['mcpet-r2-kl0.5', 'mcpet-r2-kl2', 'mcpet-r2-kl4', 'mcpet-r2-T2']
                  for s in SEEDS],
    # R8 (M5 / §4.1): merge layer 6 as a negative control, and LoRA
    'rev-r6': [(r, d, s) for r in ['fpet-r6', 'mcpet-r6'] for d in PILOT for s in SEEDS],
    'rev-lora': [(r, d, s) for r in ['lora', 'lora-fpet-r2', 'lora-mcpet-r2'] for d in PILOT for s in SEEDS],
    # pilot runs whose per-evaluation history was not kept (final-epoch accuracy, M3). Only needed if
    # the original Kaggle outputs cannot be recovered; also a reproducibility check of best_acc.
    'rev-history': [(r, d, s) for r in ['adaptformer', 'fpet-r2', 'mcpet-r2', 'fpet-r4', 'mcpet-r4']
                    for d in PILOT for s in SEEDS if s > 0 or d == 'dsprites_loc' or
                    (r, d) in [('fpet-r4', 'smallnorb_azi'), ('mcpet-r4', 'smallnorb_azi')]],
}
# one-week revision: only the runs the central criticism rests on (loss ablation, tied drop-path
# masks, validation-based checkpoint selection); everything else is reported as a limitation
# split between two people so that each pair of accounts finishes whole experiments:
# rev-week-a (friends 0, 1): no-KL and tied-mask ablations; rev-week-b (friends 2, 3): no-full-CE and val selection
PHASES['rev-week-a'] = [(r, d, s) for r in ['mcpet-r2-nokl', 'mcpet-r2-tied'] for d in PILOT for s in SEEDS]
PHASES['rev-week-b'] = [('mcpet-r2-nofullce', d, s) for d in PILOT for s in SEEDS] + PHASES['rev-val']
PHASES['rev-week'] = PHASES['rev-week-a'] + PHASES['rev-week-b']
# extra day (friends 0, 1): the rest of M1 (stochastic merging, compute-matched FPET) and M2 (no
# drop path / adapter dropout on sNORB-Azim)
PHASES['rev-day'] = ([(r, d, s) for r in ['fpet-r2-stoch', 'fpet-r2-e160'] for d in PILOT for s in SEEDS] +
                     [(r, 'smallnorb_azi', s) for r in ['fpet-r2-nodrop', 'mcpet-r2-nodrop'] for s in SEEDS])
# second session for friends 2, 3 (M5 / R14): FPET vs MC-FPET at layer 2 on the six remaining
# Structured tasks, seeds 0 and 1
PHASES['rev-struct-lite'] = [(r, d, s) for r in ['fpet-r2', 'mcpet-r2'] for d in STRUCT6 for s in [0, 1]]
# rev-day jobs that did not fit in the 11.5 h sessions (all DTD), rerun in one short session on friend 0
PHASES['rev-day-fix'] = [('fpet-r2-e160', 'dtd', 1), ('fpet-r2-e160', 'dtd', 2), ('fpet-r2-stoch', 'dtd', 2)]
# phase -> the friends it is balanced over (default: all of them)
PHASE_FRIENDS = {'rev-week-a': [0, 1], 'rev-week-b': [2, 3], 'rev-day': [0, 1], 'rev-struct-lite': [2, 3],
                 'rev-day-fix': [0]}

# job id ('run/seed<s>/dataset') -> friend; applied after the automatic split
OVERRIDES = {}

# rough T4 seconds per run: fixed (training) + per 10k test images per evaluation, from the pilot timings
COST = {'adaptformer': (2600, 100), 'fpet': (1800, 180), 'mcpet': (3500, 210), 'kd': (2500, 180),
        'lora': (2600, 100)}


def job_id(run, dataset, seed):
    return f'{run}/seed{seed}/{dataset}'


def eval_flags(run, dataset):
    if N_TEST[dataset] >= BIG_TEST:
        return '--eval_every 25' + (' --no_full_eval' if '--consistency' in RUNS[run][0] else '')
    return ''


def estimate_s(run, dataset):
    kind = run.split('-')[0] if not run.startswith('lora-') else run.split('-')[1]
    base, per_eval = COST[kind]
    epochs = RUNS[run][1]
    every = 25 if N_TEST[dataset] >= BIG_TEST else 10
    every = 16 if '--eval_every 16' in RUNS[run][0] and N_TEST[dataset] < BIG_TEST else every
    if '--consistency' in RUNS[run][0] and N_TEST[dataset] >= BIG_TEST:
        per_eval *= 0.6  # no full-branch eval
    return base * epochs / 100 + per_eval * N_TEST[dataset] / 1e4 * (epochs // every)


def assign(phases):
    """[(friend, phase, run, dataset, seed, est_s)] with each phase balanced over the friends."""
    jobs = []
    for phase in phases:
        load = [0.0] * N_FRIENDS
        friends = PHASE_FRIENDS.get(phase, range(N_FRIENDS))
        todo = sorted(PHASES[phase], key=lambda j: (-estimate_s(j[0], j[1]), job_id(*j)))
        for run, dataset, seed in todo:
            est = estimate_s(run, dataset)
            friend = min(friends, key=lambda f: (load[f], f))
            friend = OVERRIDES.get(job_id(run, dataset, seed), friend)
            load[friend] += est
            jobs.append((friend, phase, run, dataset, seed, est))
    return jobs


def command(run, dataset, seed, out, teachers):
    flags = RUNS[run][0].format(teachers=teachers, seed=seed, dataset=dataset)
    common = COMMON.replace('--method adaptformer', '') if '--method' in flags else COMMON
    return (f'python main.py --dataset {dataset} {common} --seed {seed} '
            f'--model_path {out}/{run}/seed{seed} {flags} {eval_flags(run, dataset)}').split()


def list_jobs(args):
    jobs = [j for j in assign(args.phases) if args.friend is None or j[0] == args.friend]
    for friend, phase, run, dataset, seed, est in jobs:
        print(f'friend {friend}  {phase:8s} {job_id(run, dataset, seed):40s} {est / 3600:4.1f} h')
    hours = {}
    for friend, *_, est in jobs:
        hours[friend] = hours.get(friend, 0) + est / 3600
    print(f'{len(jobs)} jobs; GPU hours per friend:', {f: round(h, 1) for f, h in sorted(hours.items())})


def run_jobs(args):
    start = time.time()
    budget = args.budget_h * 3600
    queue = [j for j in assign(args.phases) if j[0] == args.friend]
    lock = threading.Lock()
    failed = []

    def done(run, dataset, seed):
        return os.path.isfile(f'{args.out}/{run}/seed{seed}/{dataset}/summary.json')

    def zip_results():
        if args.zip:
            shutil.make_archive(args.zip[:-4] if args.zip.endswith('.zip') else args.zip, 'zip', '.', args.out)

    def worker(gpu):
        while True:
            with lock:
                job = None
                for j in queue:
                    _, _, run, dataset, seed, est = j
                    if done(run, dataset, seed):
                        continue
                    # 1.3x margin: the estimates are rough and a paired job may run slower
                    if time.time() - start + 1.3 * est > budget:
                        continue
                    job = j
                    break
                if job is None:
                    return
                queue.remove(job)
            _, phase, run, dataset, seed, est = job
            if run.startswith('kd-'):
                ckpt = TEACHER.format(teachers=args.teachers, seed=seed, dataset=dataset)
                if not os.path.isfile(ckpt):
                    print(f'[gpu{gpu}] SKIP {job_id(run, dataset, seed)}: teacher {ckpt} not found', flush=True)
                    failed.append(job_id(run, dataset, seed))
                    continue
            d = f'{args.out}/{run}/seed{seed}/{dataset}'
            os.makedirs(d, exist_ok=True)
            cmd = command(run, dataset, seed, args.out, args.teachers)
            print(f'[gpu{gpu}] start {job_id(run, dataset, seed)} (est {est / 3600:.1f} h): {" ".join(cmd)}', flush=True)
            t0 = time.time()
            with open(f'{d}/stdout.log', 'w') as log:
                code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})
            status = 'done' if code == 0 and done(run, dataset, seed) else f'FAILED (exit {code}, see {d}/stdout.log)'
            if not status == 'done':
                failed.append(job_id(run, dataset, seed))
            print(f'[gpu{gpu}] {status} {job_id(run, dataset, seed)} in {(time.time() - t0) / 3600:.2f} h', flush=True)
            with lock:
                zip_results()

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    left = [job_id(*j[2:5]) for j in assign(args.phases) if j[0] == args.friend and not done(*j[2:5])]
    print(f'\nsession finished after {(time.time() - start) / 3600:.1f} h')
    print(f'failed: {failed or "none"}')
    print(f'still to run ({len(left)}): {left or "none - all jobs for this friend are done"}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('action', choices=['list', 'run'])
    parser.add_argument('--friend', type=int, default=None, help=f'0..{N_FRIENDS - 1}')
    parser.add_argument('--phases', nargs='+', default=list(PHASES), choices=list(PHASES))
    parser.add_argument('--out', type=str, default='result/tmlr')
    parser.add_argument('--teachers', type=str, default='teachers', help='holds adaptformer/seed<s>/<dataset>.pth')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--budget_h', type=float, default=11.3, help='do not start jobs that would end after this')
    parser.add_argument('--zip', type=str, default=None, help='zip <out> here after every finished job')
    args = parser.parse_args()
    if args.action == 'run' and args.friend is None:
        parser.error('run needs --friend')
    list_jobs(args) if args.action == 'list' else run_jobs(args)
