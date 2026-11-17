"""
Statistics requested in the MC-FPET review (M3, M4, minor issues), from the saved summary.json files.
No training needed. Writes REVISION_ANALYSIS.md (and revision_analysis.json with the raw numbers).

  python3 revision_analysis.py      # reads the original 60 runs (*/pilot/), the revision runs
                                    # (results/revision/) and the final revision runs (results/revision_final/)

What it reports
  M4  task-level analysis: per-task mean gains, task-level sign test (n = 4), share of the summed gain
      from each task, mean gain without smallnorb_azi, hierarchical bootstrap over tasks and seeds,
      and a random-intercept mixed model (gain ~ 1 + (1 | task)).
  M3  final-epoch accuracy next to best-of-k for every run whose per-evaluation history is saved,
      and the number of evaluations each run had.
  minor  'recovered' as ratio of means and as mean of per-task ratios; like-for-like layer comparison;
      seed spread; FLOP reduction and training-time factor to more digits.
"""
import collections
import glob
import json
import math
import random
import statistics as st
from argparse import ArgumentParser

DATASETS = ['cifar', 'dtd', 'smallnorb_azi', 'dsprites_loc']
NAMES = {'cifar': 'CIFAR-100', 'dtd': 'DTD', 'smallnorb_azi': 'sNORB-Azim', 'dsprites_loc': 'dSpr-Loc',
         'clevr_count': 'Clevr-Count', 'clevr_dist': 'Clevr-Dist', 'dmlab': 'DMLab', 'kitti': 'KITTI-Dist',
         'dsprites_ori': 'dSpr-Ori', 'smallnorb_ele': 'sNORB-Elev'}
STRUCT6 = ['clevr_count', 'clevr_dist', 'dmlab', 'kitti', 'dsprites_ori', 'smallnorb_ele']
SEEDS = [0, 1, 2]


def load(patterns):
    runs = collections.defaultdict(dict)  # (dataset, run) -> {seed: summary}
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            j = json.load(open(f))
            runs[(j['dataset'], j['run'])][j['seed']] = j
    return runs


def binom_two_sided(k, n):
    """Exact two-sided sign test p-value for k successes out of n (p = 0.5)."""
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def t_test(x):
    from scipy import stats
    r = stats.ttest_1samp(x, 0.0)
    return r.statistic, r.pvalue


def wilcoxon(x):
    from scipy import stats
    return stats.wilcoxon(x).pvalue


def hier_bootstrap(gains, n_boot=20000, seed=0):
    """gains: {task: [per-seed gains]}. Resample tasks, then seeds within each task."""
    rng = random.Random(seed)
    tasks = list(gains)
    means = []
    for _ in range(n_boot):
        vals = []
        for t in (rng.choice(tasks) for _ in tasks):
            g = gains[t]
            vals.append(st.mean(rng.choice(g) for _ in g))
        means.append(st.mean(vals))
    means.sort()
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
    return lo, hi, sum(m <= 0 for m in means) / n_boot


def task_bootstrap(task_means, n_boot=20000, seed=0):
    """Resample the per-task mean gains only (the cross-task question)."""
    rng = random.Random(seed)
    v = list(task_means.values())
    means = sorted(st.mean(rng.choice(v) for _ in v) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def mixed_model(gains):
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    rows = [{'task': t, 'gain': g} for t, gs in gains.items() for g in gs]
    df = pd.DataFrame(rows)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = smf.mixedlm('gain ~ 1', df, groups=df['task']).fit(reml=True)
    return {'mean': res.params['Intercept'], 'se': res.bse['Intercept'], 'p': res.pvalues['Intercept'],
            'ci': tuple(res.conf_int().loc['Intercept']), 'task_var': float(res.cov_re.iloc[0, 0]),
            'resid_var': res.scale}


def fmt(x, d=2, sign=False):
    if x is None:
        return '–'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def main():
    ap = ArgumentParser()
    ap.add_argument('--glob', nargs='+', default=['*/pilot/*/seed*/*/summary.json', 'results/revision/**/summary.json'])
    ap.add_argument('--extra', nargs='+', default=['results/revision_final/**/summary.json'],
                    help='rev-final / rev-struct-rest outputs. New (run, task, seed) cells are added to the '
                         'main set; reruns of existing cells are used only for final-epoch accuracy and '
                         'histories, so the best-of-k numbers of the main tables stay those of the original runs.')
    ap.add_argument('--out', default='REVISION_ANALYSIS.md')
    args = ap.parse_args()
    runs = load(args.glob)
    reruns = load(args.extra)
    for key, by_seed in reruns.items():
        for s, j in by_seed.items():
            runs[key].setdefault(s, j)
    acc = lambda d, r: {s: j['best_acc'] for s, j in runs[(d, r)].items()}
    out, raw = [], {}
    w = out.append

    w('# Revision analysis (reviewer M3, M4 and minor issues)\n')
    w('Generated by `revision_analysis.py` from the saved `summary.json` files. '
      'Gains are MC-FPET minus FPET, same seed, best-of-k test accuracy (pp) unless stated.\n')

    # ---------------- M4 ----------------
    for r in (2, 4):
        gains = {d: [acc(d, f'mcpet-r{r}')[s] - acc(d, f'fpet-r{r}')[s] for s in SEEDS] for d in DATASETS}
        flat = [g for d in DATASETS for g in gains[d]]
        task_means = {d: st.mean(gains[d]) for d in DATASETS}
        total = sum(flat)
        t, p_t = t_test(flat)
        k_task = sum(m > 0 for m in task_means.values())
        lo, hi, p_boot = hier_bootstrap(gains)
        tlo, thi = task_bootstrap(task_means)
        mm = mixed_model(gains)
        ex = [g for d in DATASETS if d != 'smallnorb_azi' for g in gains[d]]
        raw[f'r{r}'] = {'gains': gains, 'task_means': task_means, 'mixed': mm}

        w(f'\n## M4: merge layer {r}\n')
        w('| Task | Seed gains | Task mean | Share of summed gain |')
        w('|---|---|---|---|')
        for d in DATASETS:
            share = sum(gains[d]) / total * 100 if total else float('nan')
            w(f'| {NAMES[d]} | ' + ', '.join(fmt(g, sign=True) for g in gains[d]) +
              f' | {fmt(task_means[d], sign=True)} | {share:.0f}% |')
        w('')
        w(f'- Run level (n = 12 task-seed pairs; tests seed noise within these four tasks): mean '
          f'{st.mean(flat):+.3f}, {sum(g > 0 for g in flat)}/12 positive, sign p = '
          f'{binom_two_sided(sum(g > 0 for g in flat), 12):.4f}, t({len(flat) - 1}) = {t:.2f}, p = {p_t:.4f}, '
          f'Wilcoxon p = {wilcoxon(flat):.4f}.')
        w(f'- Task level (n = 4; tests generalisation across tasks): {k_task}/4 task means positive, '
          f'two-sided sign p = {binom_two_sided(k_task, 4):.3f}; one-sample t on task means: '
          f't(3) = {t_test(list(task_means.values()))[0]:.2f}, p = {t_test(list(task_means.values()))[1]:.3f}.')
        w(f'- Without sNORB-Azim: mean {st.mean(ex):+.3f} over {len(ex)} runs, '
          f'{sum(g > 0 for g in ex)}/{len(ex)} positive (t p = {t_test(ex)[1]:.4f}).')
        w(f'- Hierarchical bootstrap (tasks, then seeds within task; 20k resamples): 95% CI '
          f'[{lo:+.2f}, {hi:+.2f}], P(mean <= 0) = {p_boot:.3f}. Bootstrap over task means only: '
          f'[{tlo:+.2f}, {thi:+.2f}].')
        if mm:
            w(f'- Mixed model gain ~ 1 + (1 | task), REML: {mm["mean"]:+.3f} ± {mm["se"]:.3f} (SE), '
              f'95% CI [{mm["ci"][0]:+.2f}, {mm["ci"][1]:+.2f}], p = {mm["p"]:.4f}; '
              f'task variance {mm["task_var"]:.3f}, residual variance {mm["resid_var"]:.3f}. '
              'With four groups the variance component is poorly estimated; read with the bootstrap.')

    # ---------------- recovered fraction ----------------
    w('\n## Fraction of the merging loss recovered (Table 4)\n')
    w('| Task | Loss r2 | Recovered r2 | Loss r4 | Recovered r4 |')
    w('|---|---|---|---|---|')
    m = lambda d, r: st.mean(acc(d, r).values())
    rec = {2: [], 4: []}
    loss_m, gain_m = {2: [], 4: []}, {2: [], 4: []}
    row = {}
    for d in DATASETS:
        cells = []
        for r in (2, 4):
            loss = m(d, 'adaptformer') - m(d, f'fpet-r{r}')
            gain = m(d, f'mcpet-r{r}') - m(d, f'fpet-r{r}')
            loss_m[r].append(loss)
            gain_m[r].append(gain)
            rec[r].append(gain / loss)
            cells += [fmt(loss), f'{gain / loss * 100:.0f}%']
        w(f'| {NAMES[d]} | ' + ' | '.join(cells) + ' |')
    for r in (2, 4):
        w(f'\n- r = {r}: ratio of means {st.mean(gain_m[r]) / st.mean(loss_m[r]) * 100:.1f}%, '
          f'mean of per-task ratios {st.mean(rec[r]) * 100:.1f}%.')
    w('- Per-task ratios with a small denominator (e.g. DTD at r = 4) are unstable and are not '
      'interpreted individually.')

    # ---------------- like-for-like ----------------
    means = {r: st.mean(m(d, r) for d in DATASETS)
             for r in ['adaptformer', 'fpet-r2', 'fpet-r4', 'mcpet-r2', 'mcpet-r4']}
    raw['means'] = means
    w('\n## Four-task means and layer comparisons\n')
    w('| Run | Mean acc |\n|---|---|')
    for r, v in means.items():
        w(f'| {r} | {v:.3f} |')
    w(f'\n- MC-FPET r2 vs FPET r4 (cross-method): {means["mcpet-r2"] - means["fpet-r4"]:+.2f}')
    w(f'- MC-FPET r2 vs MC-FPET r4 (like for like): {means["mcpet-r2"] - means["mcpet-r4"]:+.2f}')
    w(f'- FPET r2 vs FPET r4: {means["fpet-r2"] - means["fpet-r4"]:+.2f}')

    # ---------------- seed spread ----------------
    w('\n## Seed spread (sample SD over 3 seeds)\n')
    w('| Task | FPET r2 | MC-FPET r2 | FPET r4 | MC-FPET r4 |\n|---|---|---|---|---|')
    for d in DATASETS:
        w(f'| {NAMES[d]} | ' + ' | '.join(fmt(st.stdev(acc(d, r).values()))
                                          for r in ['fpet-r2', 'mcpet-r2', 'fpet-r4', 'mcpet-r4']) + ' |')

    # ---------------- M3 ----------------
    w('\n## M3: final-epoch vs best-of-k accuracy\n')
    w('Final = accuracy at the last evaluation (epoch 99), no checkpoint selection. "–" means the '
      'per-evaluation history of that run is not saved locally (only the best-of-k summary was kept).\n')
    w('Runs whose original history was not kept were rerun with the same code, seed and settings '
      '(jobs.py phase rev-final); their final accuracy is taken from the rerun, and the paired '
      'best-of-k gain on the same runs is given for comparison.\n')
    w('| Task | Run | Seed | Evals | Best-of-k (original) | Final | Best − final | Final from |\n|---|---|---|---|---|---|---|---|')
    have = collections.defaultdict(dict)
    rerun_best = collections.defaultdict(dict)
    for d in DATASETS:
        for r in ['adaptformer', 'fpet-r2', 'mcpet-r2', 'fpet-r4', 'mcpet-r4']:
            for s in SEEDS:
                j = runs[(d, r)].get(s)
                if j is None:
                    continue
                h = j.get('history') or []
                fin = j.get('final_acc', h[-1]['acc'] * 100 if h else None)
                n = j.get('n_evals', len(h) or None)
                src = 'original'
                if fin is None and s in reruns.get((d, r), {}):
                    jr = reruns[(d, r)][s]
                    fin, n, src = jr['final_acc'], jr['n_evals'], f'rerun (best {jr["best_acc"]:.2f})'
                    rerun_best[(d, r)][s] = jr['best_acc']
                else:
                    rerun_best[(d, r)][s] = j['best_acc']
                if fin is not None:
                    have[(d, r)][s] = fin
                w(f'| {NAMES[d]} | {r} | {s} | {n or "–"} | {j["best_acc"]:.2f} | {fmt(fin)} | '
                  f'{fmt(j["best_acc"] - fin) if fin is not None else "–"} | {src} |')
    w('\n### Paired gains under final-epoch accuracy (runs with both histories)\n')
    for r in (2, 4):
        g = [(d, s, have[(d, f'mcpet-r{r}')][s] - have[(d, f'fpet-r{r}')][s])
             for d in DATASETS for s in SEEDS
             if s in have[(d, f'mcpet-r{r}')] and s in have[(d, f'fpet-r{r}')]]
        gb = [acc(d, f'mcpet-r{r}')[s] - acc(d, f'fpet-r{r}')[s] for d, s, _ in g]
        gs = [rerun_best[(d, f'mcpet-r{r}')][s] - rerun_best[(d, f'fpet-r{r}')][s] for d, s, _ in g]
        if g:
            fx = [x for *_, x in g]
            tmf = [st.mean(x for dd, _, x in g if dd == d) for d in DATASETS if any(dd == d for dd, *_ in g)]
            raw[f'final_r{r}'] = {'pairs': g, 'mean': st.mean(fx), 'best_same_runs': st.mean(gs),
                                  'task_means': tmf}
            w(f'- r = {r}: {len(g)} pairs; final-epoch gain mean {st.mean(fx):+.2f} '
              f'({sum(x > 0 for x in fx)}/{len(g)} positive, t p = {t_test(fx)[1]:.4f}; task means '
              + ', '.join(fmt(x, sign=True) for x in tmf) + f') vs best-of-k gain {st.mean(gb):+.2f} on the original '
              f'runs and {st.mean(gs):+.2f} on the runs the final accuracy comes from. Pairs: '
              + ', '.join(f'{NAMES[d]} s{s} {x:+.2f}' for d, s, x in g))
        else:
            w(f'- r = {r}: no pairs with saved histories yet.')
    missing = sorted(f'{r}/seed{s}/{d}' for d in DATASETS for r in
                     ['adaptformer', 'fpet-r2', 'mcpet-r2', 'fpet-r4', 'mcpet-r4'] for s in SEEDS
                     if s not in have[(d, r)])
    w(f'\nMissing per-evaluation histories ({len(missing)} runs): ' + ', '.join(missing))

    # ---------------- M5: remaining Structured tasks (jobs.py phase rev-struct-lite) ----------------
    w('\n## M5: remaining Structured tasks, merge layer 2\n')
    st_have = [d for d in STRUCT6 if runs.get((d, 'fpet-r2')) and runs.get((d, 'mcpet-r2'))]
    if not st_have:
        w('No rev-struct runs found yet.')
    else:
        for key, label in (('best_acc', 'best-of-k'), ('final_acc', 'final epoch')):
            w(f'\n### {label}\n')
            w('| Task | Seeds | FPET | MC-FPET | Seed gains | Task mean gain |')
            w('|---|---|---|---|---|---|')
            sgains, fm, mm_ = {}, [], []
            for d in st_have:
                f, m = runs[(d, 'fpet-r2')], runs[(d, 'mcpet-r2')]
                seeds = sorted(set(f) & set(m))
                sgains[d] = [m[s][key] - f[s][key] for s in seeds]
                fm.append(st.mean(f[s][key] for s in seeds))
                mm_.append(st.mean(m[s][key] for s in seeds))
                w(f'| {NAMES[d]} | {len(seeds)} | {fm[-1]:.2f} | {mm_[-1]:.2f} | ' +
                  ', '.join(fmt(g, sign=True) for g in sgains[d]) + f' | {fmt(st.mean(sgains[d]), sign=True)} |')
            flat = [g for d in st_have for g in sgains[d]]
            tm = [st.mean(sgains[d]) for d in st_have]
            n, k = len(flat), sum(g > 0 for g in flat)
            t, p = t_test(flat)
            tt, tp = t_test(tm)
            lo, hi, pb = hier_bootstrap(sgains)
            w(f'\n- Mean over {len(st_have)} tasks: FPET {st.mean(fm):.2f}, MC-FPET {st.mean(mm_):.2f}, '
              f'gain {st.mean(mm_) - st.mean(fm):+.2f}.')
            w(f'- Run level: {k}/{n} pairs positive (sign p = {binom_two_sided(k, n):.4f}), '
              f't({n - 1}) = {t:.2f}, p = {p:.4f}.')
            w(f'- Task level: {sum(x > 0 for x in tm)}/{len(tm)} task means positive '
              f'(sign p = {binom_two_sided(sum(x > 0 for x in tm), len(tm)):.3f}), '
              f't({len(tm) - 1}) = {tt:.2f}, p = {tp:.4f}.')
            w(f'- Hierarchical bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}], P(mean <= 0) = {pb:.3f}.')
            raw[f'struct6_{key}'] = {'gains': sgains, 'fpet_mean': st.mean(fm), 'mc_mean': st.mean(mm_),
                                     'fpet': dict(zip(st_have, fm)), 'mc': dict(zip(st_have, mm_)),
                                     'run_t': (t, p), 'task_t': (tt, tp), 'k': k, 'n': n, 'boot': (lo, hi),
                                     'k_task': sum(x > 0 for x in tm)}
            af = [st.mean(j[key] for j in runs[(d, 'adaptformer')].values()) if runs.get((d, 'adaptformer')) else None
                  for d in st_have]
            if all(a is not None for a in af):
                w('- AdaptFormer (no merging): ' + ', '.join(f'{NAMES[d]} {a:.2f}' for d, a in zip(st_have, af)) +
                  f'; mean {st.mean(af):.2f}. Merging loss {st.mean(af) - st.mean(fm):.2f}, recovered by MC-FPET '
                  f'{(st.mean(mm_) - st.mean(fm)) / (st.mean(af) - st.mean(fm)) * 100:.0f}% (ratio of means). '
                  'Per task: ' + ', '.join(
                      f'{NAMES[d]} {(b - a) / (c - a) * 100:.0f}%' if c - a > 0.3 else f'{NAMES[d]} (loss {c - a:+.2f}, n/a)'
                      for d, a, b, c in zip(st_have, fm, mm_, af)))
                raw[f'struct6_{key}']['adaptformer'] = dict(zip(st_have, af))
            raw[f'struct6_{key}']['sd'] = {
                r: {d: st.stdev(j[key] for j in runs[(d, r)].values()) for d in st_have}
                for r in ('adaptformer', 'fpet-r2', 'mcpet-r2') if all(runs.get((d, r)) for d in st_have)}
        w('\n### Full-token path (best-of-k; – where the test set is too large for full-path evaluation)\n')
        w('| Task | FPET merged | FPET full path | MC-FPET merged | MC-FPET full path |')
        w('|---|---|---|---|---|')
        for d in st_have:
            f, m = runs[(d, 'fpet-r2')], runs[(d, 'mcpet-r2')]
            avg = lambda r, k: (None if any(r[s].get(k) is None for s in r) else st.mean(r[s][k] for s in r))
            w(f'| {NAMES[d]} | {fmt(avg(f, "best_acc"))} | {fmt(avg(f, "best_acc_full_branch"))} | '
              f'{fmt(avg(m, "best_acc"))} | {fmt(avg(m, "best_acc_full_branch"))} |')
        tf = st.mean(runs[(d, 'fpet-r2')][s]['total_train_time_s'] for d in st_have for s in runs[(d, 'fpet-r2')])
        tmc = st.mean(runs[(d, 'mcpet-r2')][s]['total_train_time_s'] for d in st_have for s in runs[(d, 'mcpet-r2')])
        w(f'\n- Mean training time per run: FPET {tf / 3600:.2f} h, MC-FPET {tmc / 3600:.2f} h ({tmc / tf:.2f}x).')

    # ---------------- M1 / M2 / M6: ablations at layer 2 (Table tab:ablation) ----------------
    ABL = [('fpet-r2', 'FPET'), ('mcpet-r2', 'MC-FPET'), ('mcpet-r2-nokl', 'No KL'),
           ('mcpet-r2-nofullce', 'No full-token CE'), ('fpet-r2-stoch', 'Stochastic merging (p=0.5)'),
           ('fpet-r2-e160', 'FPET, 160 epochs'), ('mcpet-r2-tied', 'MC-FPET, tied masks'),
           ('kd-r2', 'External AdaptFormer teacher')]
    w('\n## M1 / M2 / M6: ablations, merge layer 2 (best-of-k, mean over seeds)\n')
    w('Gain = paired difference to FPET r2 (same seed) on the four-task mean; only rows with all '
      '4 tasks x 3 seeds get a mean and a gain.\n')
    w('| Run | ' + ' | '.join(NAMES[d] for d in DATASETS) + ' | Mean | vs FPET | vs MC-FPET | n |')
    w('|---|' + '---|' * (len(DATASETS) + 4))
    base = {s: st.mean(acc(d, 'fpet-r2')[s] for d in DATASETS) for s in SEEDS}
    mcb = {s: st.mean(acc(d, 'mcpet-r2')[s] for d in DATASETS) for s in SEEDS}
    for r, label in ABL:
        have = {d: acc(d, r) if (d, r) in runs else {} for d in DATASETS}
        cells = [fmt(st.mean(have[d].values())) if have[d] else '–' for d in DATASETS]
        n = sum(len(have[d]) for d in DATASETS)
        full = all(set(have[d]) >= set(SEEDS) for d in DATASETS)
        if full:
            per_seed = {s: st.mean(have[d][s] for d in DATASETS) for s in SEEDS}
            mean = st.mean(per_seed.values())
            g = [per_seed[s] - base[s] for s in SEEDS]
            gm = [per_seed[s] - mcb[s] for s in SEEDS]
            raw[f'ablation_{r}'] = {'per_task': {d: st.mean(have[d].values()) for d in DATASETS}, 'mean': mean,
                                    'gain_vs_fpet_per_seed': g, 'gain_vs_mc_per_seed': gm}
            w(f'| {label} | ' + ' | '.join(cells) + f' | {mean:.2f} | {fmt(st.mean(g), sign=True)} '
              f'(seeds {", ".join(fmt(x, sign=True) for x in g)}) | {fmt(st.mean(gm), sign=True)} | {n} |')
        else:
            w(f'| {label} | ' + ' | '.join(cells) + f' | – | – | – | {n} |')
    # fraction of the MC-FPET gain each ablation keeps
    for r, label in ABL[2:]:
        if f'ablation_{r}' in raw:
            keep = (raw[f'ablation_{r}']['mean'] - st.mean(base.values())) / (st.mean(mcb.values()) - st.mean(base.values()))
            w(f'- {label}: keeps {keep * 100:.0f}% of the MC-FPET gain over FPET.')
    # per-task paired gains vs FPET, to see where each ablation loses
    w('\nPer-task paired gain vs FPET r2 (mean over seeds):\n')
    w('| Run | ' + ' | '.join(NAMES[d] for d in DATASETS) + ' |')
    w('|---|' + '---|' * len(DATASETS))
    for r, label in ABL[1:]:
        if all((d, r) in runs for d in DATASETS):
            w(f'| {label} | ' + ' | '.join(
                fmt(st.mean(acc(d, r)[s] - acc(d, 'fpet-r2')[s] for s in acc(d, r)), sign=True) for d in DATASETS) + ' |')

    # M2: no stochastic regularisation on sNORB-Azim
    d = 'smallnorb_azi'
    if (d, 'fpet-r2-nodrop') in runs and (d, 'mcpet-r2-nodrop') in runs:
        f, m = acc(d, 'fpet-r2-nodrop'), acc(d, 'mcpet-r2-nodrop')
        seeds = sorted(set(f) & set(m))
        g = [m[s] - f[s] for s in seeds]
        w(f'\n### M2: sNORB-Azim without drop path / adapter dropout ({len(seeds)} seeds)\n')
        w(f'- FPET {st.mean(f[s] for s in seeds):.2f}, MC-FPET {st.mean(m[s] for s in seeds):.2f}, '
          f'paired gain {st.mean(g):+.2f} (seeds {", ".join(fmt(x, sign=True) for x in g)}); '
          f'with stochastic regularisation the gain is {st.mean(acc(d, "mcpet-r2")[s] - acc(d, "fpet-r2")[s] for s in SEEDS):+.2f}.')
        raw['nodrop'] = {'fpet': st.mean(f[s] for s in seeds), 'mc': st.mean(m[s] for s in seeds), 'gains': g}

    # M3: validation-based checkpoint selection (train800, select on val200)
    if all((d, r) in runs for d in DATASETS for r in ('fpet-r2-val', 'mcpet-r2-val')):
        w('\n### M3: checkpoint selected on val200 (trained on train800 only, seed 0)\n')
        w('| Task | FPET val-selected | MC-FPET val-selected | Gain | FPET best-of-k (same run) | MC-FPET best-of-k (same run) | Gain |')
        w('|---|---|---|---|---|---|---|')
        vs, bk = [], []
        for d in DATASETS:
            f, m = runs[(d, 'fpet-r2-val')][0], runs[(d, 'mcpet-r2-val')][0]
            vs.append(m['val_selected_test_acc'] - f['val_selected_test_acc'])
            bk.append(m['best_acc'] - f['best_acc'])
            w(f'| {NAMES[d]} | {f["val_selected_test_acc"]:.2f} | {m["val_selected_test_acc"]:.2f} | {vs[-1]:+.2f} | '
              f'{f["best_acc"]:.2f} | {m["best_acc"]:.2f} | {bk[-1]:+.2f} |')
        fv = st.mean(runs[(d, 'fpet-r2-val')][0]['val_selected_test_acc'] for d in DATASETS)
        mv = st.mean(runs[(d, 'mcpet-r2-val')][0]['val_selected_test_acc'] for d in DATASETS)
        w(f'\n- Four-task mean, val-selected: FPET {fv:.2f}, MC-FPET {mv:.2f}, gain {st.mean(vs):+.2f} '
          f'({sum(x > 0 for x in vs)}/4 tasks positive); best-of-k on the same runs: gain {st.mean(bk):+.2f}.')
        raw['val_select'] = {'gains_val': vs, 'gains_bestk': bk, 'fpet': fv, 'mc': mv}

    # ---------------- rev-final: loss weights, cheaper training, LoRA, layer 6, pooled ----------------
    def cell_mean(r, d):
        return st.mean(acc(d, r).values())

    def paired(r, ref, tasks=DATASETS):
        """Per-seed paired gains of run r over ref on every (task, seed) both have."""
        return {d: [acc(d, r)[s] - acc(d, ref)[s] for s in SEEDS if s in acc(d, r) and s in acc(d, ref)]
                for d in tasks if (d, r) in runs and (d, ref) in runs}

    def runinfo(r, k, tasks=DATASETS):
        v = [j[k] for d in tasks for j in runs.get((d, r), {}).values() if j.get(k)]
        return st.mean(v) if v else None

    w('\n## Loss weights on CIFAR-100, merge layer 2 (reviewer minor point)\n')
    w('| Setting | Seed accuracies | Mean | Paired vs MC-FPET (λkl = 1, T = 1) | Paired vs FPET |')
    w('|---|---|---|---|---|')
    sweep = {}
    for r, label in [('mcpet-r2-kl0.5', 'λkl = 0.5'), ('mcpet-r2', 'λkl = 1 (main)'), ('mcpet-r2-kl2', 'λkl = 2'),
                     ('mcpet-r2-kl4', 'λkl = 4'), ('mcpet-r2-T2', 'T = 2, λkl = 1')]:
        if ('cifar', r) not in runs:
            continue
        a = acc('cifar', r)
        gm = paired(r, 'mcpet-r2', ['cifar']).get('cifar', [])
        gf = paired(r, 'fpet-r2', ['cifar']).get('cifar', [])
        sweep[r] = {'mean': st.mean(a.values()), 'vs_mc': gm, 'vs_fpet': gf}
        w(f'| {label} | ' + ', '.join(f'{a[s]:.2f}' for s in sorted(a)) + f' | {st.mean(a.values()):.2f} | '
          + (', '.join(fmt(x, sign=True) for x in gm) + f' (mean {st.mean(gm):+.2f})' if r != 'mcpet-r2' else '–')
          + f' | {st.mean(gf):+.2f} |')
    raw['sweep_cifar'] = sweep

    if all((d, 'mcpet-r2-every2') in runs for d in DATASETS):
        r = 'mcpet-r2-every2'
        g = paired(r, 'fpet-r2')
        gmc = paired('mcpet-r2', 'fpet-r2')
        per_seed = {s: st.mean(acc(d, r)[s] for d in DATASETS) for s in SEEDS}
        mean = st.mean(per_seed.values())
        flat = [x for d in DATASETS for x in g[d]]
        keep = st.mean(flat) / st.mean(x for d in DATASETS for x in gmc[d])
        t_mc = runinfo('mcpet-r2', 'total_train_time_s')
        t_e2 = runinfo(r, 'total_train_time_s')
        # same-session comparison: MC-FPET reruns on the accounts that also ran every2
        w('\n## Cheaper training: full-token path on every second step (merge layer 2)\n')
        w('| Task | FPET | Every 2nd step | MC-FPET | Gain every2 | Gain MC-FPET |\n|---|---|---|---|---|---|')
        for d in DATASETS:
            w(f'| {NAMES[d]} | {cell_mean("fpet-r2", d):.2f} | {cell_mean(r, d):.2f} | {cell_mean("mcpet-r2", d):.2f} | '
              f'{st.mean(g[d]):+.2f} | {st.mean(gmc[d]):+.2f} |')
        w(f'\n- Four-task mean {mean:.2f} (FPET {st.mean(base.values()):.2f}, MC-FPET {st.mean(mcb.values()):.2f}); '
          f'paired gain over FPET {st.mean(flat):+.2f} ({sum(x > 0 for x in flat)}/12 positive, t p = {t_test(flat)[1]:.4f}), '
          f'keeps {keep * 100:.0f}% of the MC-FPET gain.')
        w(f'- Peak memory {runinfo(r, "peak_train_mem_gb"):.2f} GB (unchanged: every step that runs the full path '
          f'stores its activations). Mean run time {t_e2 / 3600:.2f} h against {t_mc / 3600:.2f} h for MC-FPET '
          f'({(1 - t_e2 / t_mc) * 100:.0f}% less) and {runinfo("fpet-r2", "total_train_time_s") / 3600:.2f} h for FPET.')
        raw['every2'] = {'mean': mean, 'gain': st.mean(flat), 'per_task': {d: st.mean(g[d]) for d in DATASETS},
                         'keep': keep, 'mem': runinfo(r, 'peak_train_mem_gb'), 'time_h': t_e2 / 3600,
                         'time_mc_h': t_mc / 3600, 'k': sum(x > 0 for x in flat), 'p': t_test(flat)[1]}

    if all((d, 'kd-r2') in runs for d in DATASETS):
        g = paired('kd-r2', 'fpet-r2')
        nat = [x for d in ('cifar', 'dtd') for x in g[d]]
        stru = [x for d in ('smallnorb_azi', 'dsprites_loc') for x in g[d]]
        w('\n## External teacher (kd-r2), per-seed gains over FPET\n')
        for d in DATASETS:
            w(f'- {NAMES[d]}: ' + ', '.join(fmt(x, sign=True) for x in g[d]) + f' (mean {st.mean(g[d]):+.2f}); '
              'MC-FPET ' + ', '.join(fmt(x, sign=True) for x in paired("mcpet-r2", "fpet-r2")[d]))
        w(f'- Natural tasks: mean {st.mean(nat):+.2f} (range {min(nat):+.2f} to {max(nat):+.2f}); Structured tasks: '
          f'mean {st.mean(stru):+.2f} (range {min(stru):+.2f} to {max(stru):+.2f}).')
        kg = paired('kd-r2', 'mcpet-r2')
        flat = [x for d in DATASETS for x in kg[d]]
        w(f'- External teacher vs MC-FPET: {sum(x > 0 for x in flat)}/12 pairs better, mean {st.mean(flat):+.2f}.')
        w(f'- Peak memory {runinfo("kd-r2", "peak_train_mem_gb"):.2f} GB (frozen teacher in eval mode, no grad); '
          f'the teacher needs its own AdaptFormer training run first ({runinfo("adaptformer", "total_train_time_s") / 3600:.2f} h).')
        raw['kd'] = {'per_task': {d: g[d] for d in DATASETS}, 'natural': st.mean(nat), 'structured': st.mean(stru),
                     'nat_range': (min(nat), max(nat)), 'struct_range': (min(stru), max(stru)),
                     'vs_mc_better': sum(x > 0 for x in flat), 'mem': runinfo('kd-r2', 'peak_train_mem_gb')}

    for tag, (ref_un, f_run, m_run), title in [
            ('lora', ('lora', 'lora-fpet-r2', 'lora-mcpet-r2'), 'LoRA (rank 8), merge layer 2'),
            ('r6', ('adaptformer', 'fpet-r6', 'mcpet-r6'), 'AdaptFormer, merge layer 6 (negative control)')]:
        if not all((d, x) in runs for d in DATASETS for x in (ref_un, f_run, m_run)):
            continue
        g = paired(m_run, f_run)
        flat = [x for d in DATASETS for x in g[d]]
        tm = [st.mean(g[d]) for d in DATASETS]
        un = [cell_mean(ref_un, d) for d in DATASETS]
        fm = [cell_mean(f_run, d) for d in DATASETS]
        mm_ = [cell_mean(m_run, d) for d in DATASETS]
        loss = st.mean(un) - st.mean(fm)
        w(f'\n## {title}\n')
        w(f'| Task | {ref_un} (no merging) | {f_run} | {m_run} | Seed gains | Mean gain |\n|---|---|---|---|---|---|')
        for i, d in enumerate(DATASETS):
            w(f'| {NAMES[d]} | {un[i]:.2f} | {fm[i]:.2f} | {mm_[i]:.2f} | ' + ', '.join(fmt(x, sign=True) for x in g[d])
              + f' | {tm[i]:+.2f} |')
        w(f'| Mean | {st.mean(un):.2f} | {st.mean(fm):.2f} | {st.mean(mm_):.2f} | | {st.mean(flat):+.2f} |')
        w(f'\n- Run level {sum(x > 0 for x in flat)}/12 positive, t p = {t_test(flat)[1]:.4f}; task level '
          f'{sum(x > 0 for x in tm)}/4 positive, t p = {t_test(tm)[1]:.3f}. Merging loss {loss:.2f}, recovered '
          f'{st.mean(flat) / loss * 100:.0f}% (ratio of means).')
        w(f'- Peak memory (GB): {ref_un} {runinfo(ref_un, "peak_train_mem_gb"):.2f}, {f_run} '
          f'{runinfo(f_run, "peak_train_mem_gb"):.2f}, {m_run} {runinfo(m_run, "peak_train_mem_gb"):.2f}. '
          f'Inference GFLOPs {runinfo(f_run, "gflops"):.2f} vs {runinfo(ref_un, "gflops"):.2f} '
          f'({(1 - runinfo(f_run, "gflops") / runinfo(ref_un, "gflops")) * 100:.1f}% fewer). Run time (h): '
          f'{f_run} {runinfo(f_run, "total_train_time_s") / 3600:.2f}, {m_run} {runinfo(m_run, "total_train_time_s") / 3600:.2f}.')
        raw[tag] = {'unmerged': dict(zip(DATASETS, un)), 'fpet': dict(zip(DATASETS, fm)), 'mc': dict(zip(DATASETS, mm_)),
                    'gains': g, 'mean_gain': st.mean(flat), 'k': sum(x > 0 for x in flat), 'p_run': t_test(flat)[1],
                    'p_task': t_test(tm)[1], 'loss': loss, 'means': (st.mean(un), st.mean(fm), st.mean(mm_)),
                    'mem': {x: runinfo(x, 'peak_train_mem_gb') for x in (ref_un, f_run, m_run)},
                    'gflops': runinfo(f_run, 'gflops')}

    # pooled: the four main tasks plus the six other Structured tasks, merge layer 2
    allt = DATASETS + STRUCT6
    if all((d, r) in runs for d in allt for r in ('fpet-r2', 'mcpet-r2')):
        g = paired('mcpet-r2', 'fpet-r2', allt)
        flat = [x for d in allt for x in g[d]]
        tm = [st.mean(g[d]) for d in allt]
        mm = mixed_model(g)
        lo, hi, pb = hier_bootstrap(g)
        k = sum(x > 0 for x in tm)
        w('\n## Pooled: 10 tasks (4 main + 6 other Structured), merge layer 2\n')
        w(f'- {len(flat)} task-seed pairs, mean gain {st.mean(flat):+.2f}, {sum(x > 0 for x in flat)}/{len(flat)} positive, '
          f't p = {t_test(flat)[1]:.2e}.')
        w(f'- Task level: {k}/10 task means positive (sign p = {binom_two_sided(k, 10):.4f}), '
          f't(9) = {t_test(tm)[0]:.2f}, p = {t_test(tm)[1]:.4f}; Wilcoxon over task means p = {wilcoxon(tm):.4f}.')
        w(f'- Hierarchical bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}].')
        if mm:
            w(f'- Mixed model: {mm["mean"]:+.2f} ± {mm["se"]:.2f}, 95% CI [{mm["ci"][0]:+.2f}, {mm["ci"][1]:+.2f}], p = {mm["p"]:.4f}.')
        raw['pooled10'] = {'mean': st.mean(flat), 'k_pairs': sum(x > 0 for x in flat), 'n_pairs': len(flat),
                           'p_run': t_test(flat)[1], 'k_task': k, 'p_sign_task': binom_two_sided(k, 10),
                           't_task': t_test(tm), 'p_wilcoxon_task': wilcoxon(tm), 'boot': (lo, hi), 'mixed': mm}

    # ---------------- cost numbers to more digits ----------------
    w('\n## Cost numbers (unrounded)\n')
    info = collections.defaultdict(list)
    for (d, r), v in runs.items():
        info[r] += v.values()
    g = {r: st.mean(j['gflops'] for j in info[r]) for r in info}
    w(f'- FLOP reduction r2: {(1 - g["fpet-r2"] / g["adaptformer"]) * 100:.2f}%, '
      f'r4: {(1 - g["fpet-r4"] / g["adaptformer"]) * 100:.2f}% '
      f'(GFLOPs {g["adaptformer"]:.3f} / {g["fpet-r2"]:.3f} / {g["fpet-r4"]:.3f}).')
    mem = {r: st.mean(j['peak_train_mem_gb'] for j in info[r] if j.get('peak_train_mem_gb')) for r in info}
    w('- Peak training memory (GB): ' + ', '.join(f'{r} {v:.2f}' for r, v in sorted(mem.items())) +
      f'; MC-FPET r2 / FPET r2 = {mem["mcpet-r2"] / mem["fpet-r2"]:.2f}x, '
      f'MC-FPET r2 / AdaptFormer = {mem["mcpet-r2"] / mem["adaptformer"]:.2f}x.')

    open(args.out, 'w').write('\n'.join(out) + '\n')
    json.dump(raw, open(args.out.replace('.md', '.json').replace('REVISION_ANALYSIS', 'revision_analysis'), 'w'),
              indent=1, default=str)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
