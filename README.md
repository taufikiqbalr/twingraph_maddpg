# TwinGraph-MADDPG: experiment settings and numerical reproduction

This repository accompanies the TwinGraph-MADDPG manuscript. It provides the
experiment settings, recorded observations, and executable analysis and plotting
code for the six-method PPP and T-Drive comparisons.

## Experiment coverage

- Six methods: TwinGraph-MADDPG, MADDPG, MATD3, MAPPO, GAT-MADDPG, and centralized DDQN.
- Two mobility settings: PPP and T-Drive; four deployment sizes: 3, 5, 7, and 9 agents.
- Three training seeds: 11, 17, and 23; 1,000 episodes of 80 steps per trained policy.
- 144 trained policies and 144,000 recorded training episodes.
- 2,880 primary evaluation episodes: 20 scenarios per trained policy, with held-out vehicles for every T-Drive scale.
- 1,440 additional T-Drive training-partition diagnostic episodes, analyzed separately.


## Quick start

Use Python 3.12 for the tested plotting environment. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/analyze_results.py
python scripts/plot_results.py
```

The analysis and integrity checks require only the Python standard library.
Plotting uses the pinned libraries in `requirements.txt`. All input files are
bundled, so these commands need no private repository, dataset download, model,
GPU, network service, or credentials. Outputs are written to `outputs/` and are
excluded from version control. Each script also accepts `--help`.

## What is included

| Location | Content |
|---|---|
| `configs/models/` | Six model descriptions and reported architecture/learning settings |
| `configs/conditions.json` | Resolved public settings for all 48 conditions |
| `configs/deployments.json` | Deployment sizes and environment settings |
| `configs/protocol.json` | Training/evaluation protocol and scenario schedules |
| `configs/run_manifest.csv` | Per-run settings, seeds, hardware and recorded durations |
| `data/evaluation_episodes.csv` | All 4,320 evaluation records, with primary/diagnostic split labels |
| `data/training_episodes.csv.gz` | All 144,000 training observations for reward, lookup cost and fetches |
| `scripts/analyze_results.py` | Seed-level aggregation, uncertainty and paired comparisons |
| `scripts/plot_results.py` | Eight comparison figures with editable SVG, vector PDF and 600-dpi PNG outputs |
| `scripts/verify_release.py` | File-integrity and dataset-coverage checks |

Learning rates, batch sizes, hidden-layer counts and dimensions, target-update
rates, seed counts, episode budgets and training durations are documented in
[HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md). The
[training protocol](docs/TRAINING_PROTOCOL.md) describes clocks, scenario pairing,
trace partitions and timing scopes. See
[metrics and statistics](docs/METRICS_AND_STATISTICS.md) for metric definitions
and the exact aggregation and hypothesis-test procedures.

## Outputs and interpretation

The analysis produces 216 seed-mean records, 720 metric summaries and 600 paired
contrasts in `outputs/tables/`. The primary lookup-cost tests use two separate
Holm families of 20 comparisons, one for each mobility setting. The T-Drive
training-partition diagnostic forms its own family. Three training seeds limit
statistical resolution; favorable means do not establish statistical superiority.

The plotter produces two final-policy figures (PPP and held-out T-Drive lookup,
communication and energy) and six training figures (reward, lookup cost and
cross-shard fetches for each mobility setting). It also exports every plotted
mean and sample SD. Training curves use the trailing 20-episode mean within
each seed, followed by the mean and sample SD across the three seeds. The bands
are not confidence intervals. Reward and fetch axes match between PPP and
T-Drive. The output PNGs default to 600 dpi; `--dpi 150` creates smaller previews.

Metrics retain the published global-cache scope. Lookup, communication and
energy are event-based simulator proxies. The training `reward` column is the
global agent's episodic return, distinct from the sum of returns across all
agents (`network_reward` in evaluation data). Deployment sizes also change
population, infrastructure and area, so panels do not isolate agent count alone.
