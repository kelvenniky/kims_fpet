# Per-run results

Each run folder holds `summary.json` (accuracy at every evaluation, best and final-epoch accuracy,
full-token accuracy where evaluated, GFLOPs, latency, parameter counts, peak training memory, training
time) and, where it was saved, `run_info.json` (all command-line arguments, software environment, GPU
and the Git commit of the code). Layout: `<run>/seed<seed>/<dataset>/`. Run names are those of `jobs.py`.

| Folder | Runs | What |
|---|---|---|
| `../*/pilot/` | 60 | Original study: AdaptFormer, FPET and MC-FPET at merge layers 2 and 4, four tasks, seeds 0-2 |
| `revision/` | 98 | Loss-term ablations, stochastic merging, 160-epoch FPET, tied masks, no-dropout runs, validation-selected runs, six further Structured tasks (seeds 0-1) |
| `revision_final/` | 176 | External teacher, every-second-step, loss-weight sweep, merge layer 6, LoRA, six Structured tasks (seed 2 and AdaptFormer), and repeats of original runs whose per-evaluation history had not been kept |

The repeats in `revision_final/` are used only for final-epoch accuracy and training curves; the
best-of-k accuracies in the paper's main tables are those of the original runs. Checkpoints are not
included.

Every statistic in the paper is produced by `python3 revision_analysis.py` (writes `REVISION_ANALYSIS.md`),
and every figure by `python3 paper/scripts/make_figures.py`, both run from the repository root.
