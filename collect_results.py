"""
Collects the summary.json written by every run under --root and writes
  <root>/all_runs.csv   one row per (run, seed, dataset): raw evidence
  <root>/report.md      accuracy table, efficiency table and paired comparisons

Expected layout: <root>/<run>/[seed<k>/]<dataset>/summary.json
"""
import os
import re
import csv
import json
import glob
import statistics
from argparse import ArgumentParser

NATURAL = ['cifar', 'caltech101', 'dtd', 'oxford_flowers102', 'oxford_iiit_pet', 'svhn', 'sun397']
SPECIALIZED = ['patch_camelyon', 'eurosat', 'resisc45', 'diabetic_retinopathy']
STRUCTURED = ['clevr_count', 'clevr_dist', 'dmlab', 'kitti', 'dsprites_loc', 'dsprites_ori',
              'smallnorb_azi', 'smallnorb_ele']
ORDER = NATURAL + SPECIALIZED + STRUCTURED


def load_runs(root):
    rows = []
    for path in glob.glob(os.path.join(root, '**', 'summary.json'), recursive=True):
        with open(path) as f:
            s = json.load(f)
        run_dir = os.path.dirname(os.path.dirname(path))
        m = re.fullmatch(r'seed(\d+)', os.path.basename(run_dir))
        if m:
            run_dir = os.path.dirname(run_dir)
        s['run'] = os.path.relpath(run_dir, root)
        s['path'] = path
        rows.append(s)
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def main():
    parser = ArgumentParser()
    parser.add_argument('--root', default='./result/pilot')
    parser.add_argument('--pairs', nargs='*', default=[],
                        help='baseline:candidate run names to compare, e.g. fpet-r2:mcpet-r2')
    args = parser.parse_args()

    rows = load_runs(args.root)
    if not rows:
        raise SystemExit(f'no summary.json found under {args.root}')

    fields = ['run', 'seed', 'dataset', 'method', 'r_layer', 'consistency', 'best_acc', 'best_epoch',
              'best_acc_full_branch', 'gflops', 'latency_ms_per_img', 'peak_train_mem_gb',
              'total_train_time_s', 'params_trainable', 'path']
    with open(os.path.join(args.root, 'all_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r['run'], r['seed'], r['dataset'])):
            w.writerow(r)

    # (run, dataset) -> list over seeds
    acc, acc_full = {}, {}
    for r in rows:
        acc.setdefault((r['run'], r['dataset']), []).append(r['best_acc'])
        acc_full.setdefault((r['run'], r['dataset']), []).append(r['best_acc_full_branch'])
    runs = sorted({r['run'] for r in rows})
    datasets = [d for d in ORDER if any((run, d) in acc for run in runs)]
    datasets += sorted({r['dataset'] for r in rows} - set(datasets))
    complete = set(datasets) >= set(ORDER)

    def cell(run, d):
        xs = acc.get((run, d))
        if not xs:
            return '-'
        if len(xs) == 1:
            return fmt(xs[0], 1)
        return f'{statistics.mean(xs):.1f}±{statistics.stdev(xs):.1f}'

    def run_mean(run, ds):
        vals = [statistics.mean(acc[(run, d)]) for d in ds if (run, d) in acc]
        return mean(vals) if len(vals) == len(ds) else None

    lines = [f'# Results under `{args.root}`', '',
             f'{len(rows)} runs found. Accuracy = best top-1 test accuracy (%), mean±std over seeds when >1 seed.', '',
             '## Accuracy', '']
    header = ['run'] + datasets + ['Mean']
    if complete:
        header += ['Natural', 'Specialized', 'Structured', 'VTAB Avg']
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '---|' * len(header))
    for run in runs:
        row = [run] + [cell(run, d) for d in datasets] + [fmt(run_mean(run, datasets))]
        if complete:
            groups = [run_mean(run, g) for g in (NATURAL, SPECIALIZED, STRUCTURED)]
            row += [fmt(g) for g in groups] + [fmt(mean(groups) if None not in groups else None)]
        lines.append('| ' + ' | '.join(row) + ' |')

    lines += ['', '## Efficiency and full-token branch', '',
              'GFLOPs and latency are inference costs of the deployed (merged) model; latency is ms/image at batch 256 '
              'and only comparable across runs on the same GPU. Peak memory and train time are for training. '
              '"Full-branch acc" is the same weights evaluated without merging.', '',
              '| run | GFLOPs | latency (ms/img) | peak train mem (GB) | train time (min) | trainable params (M) | mean acc | mean full-branch acc |',
              '|---|---|---|---|---|---|---|---|']
    for run in runs:
        rr = [r for r in rows if r['run'] == run]
        tt = mean([r.get('total_train_time_s') for r in rr])
        pt = mean([r['params_trainable'] for r in rr])
        lines.append('| ' + ' | '.join([
            run, fmt(mean([r['gflops'] for r in rr])), fmt(mean([r['latency_ms_per_img'] for r in rr]), 3),
            fmt(mean([r.get('peak_train_mem_gb') for r in rr])), fmt(tt / 60 if tt else None, 1),
            fmt(pt / 1e6 if pt else None, 3), fmt(run_mean(run, datasets)),
            fmt(mean([mean(acc_full[(run, d)]) for d in datasets if (run, d) in acc_full])),
        ]) + ' |')

    for pair in args.pairs:
        base, cand = pair.split(':')
        common = [d for d in datasets if (base, d) in acc and (cand, d) in acc]
        lines += ['', f'## {cand} vs {base}', '']
        if not common:
            lines.append('No datasets in common.')
            continue
        lines += ['| dataset | ' + base + ' | ' + cand + ' | Δ |', '|---|---|---|---|']
        deltas = []
        for d in common:
            b, c = statistics.mean(acc[(base, d)]), statistics.mean(acc[(cand, d)])
            deltas.append(c - b)
            lines.append(f'| {d} | {b:.2f} | {c:.2f} | {c - b:+.2f} |')
        wins = sum(x > 0 for x in deltas)
        ties = sum(x == 0 for x in deltas)
        lines += ['', f'Mean Δ = {statistics.mean(deltas):+.2f} over {len(common)} datasets; '
                      f'{cand} wins {wins}, ties {ties}, loses {len(deltas) - wins - ties}.']

    with open(os.path.join(args.root, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
