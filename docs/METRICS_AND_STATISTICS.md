# Metrics and statistical analysis

This release reproduces numerical analyses of recorded evaluation episodes. It does not rerun policy training or the simulator. The environment and learning implementations are not included in this limited release. The script therefore validates the identity and completeness of the released records, but cannot independently verify a model checkpoint, reconstruct a trajectory, or confirm a recorded metric against the simulator.

## Run the analysis

Python 3.9 or later is sufficient; the script uses only the standard library.

```bash
python scripts/analyze_results.py
```

The default data and output locations are resolved relative to the release directory, so invoking the script by its absolute path also works from another working directory. Explicit relative arguments are resolved relative to the current working directory:

```bash
python scripts/analyze_results.py --data data/evaluation_episodes.csv --output outputs/tables
```

The command writes three CSV files:

| File | Rows | Unit |
|---|---:|---|
| `seed_means.csv` | 216 | One trained policy seed within a mobility, deployment scale, and evaluation split |
| `metric_summary.csv` | 720 | One metric and policy within a mobility, scale, and split |
| `contrasts.csv` | 600 | One metric and TwinGraph-versus-baseline comparison within a mobility, scale, and split |

## Data and evaluation scope

All records identify the final checkpoint after **1,000 training episodes**. There are six policies, four deployment scales (3, 5, 7, and 9 agents), and three training seeds (11, 17, and 23). Each trained policy is evaluated on 20 scenarios per evaluation split. Evaluation uses one logical second per environment step. Training and evaluation episodes have 80 environment steps in the recorded protocol.

The primary results are the rows with `split=eval`: 1,440 PPP episodes on fresh synthetic scenarios and 1,440 T-Drive episodes on held-out vehicle identities, for 2,880 primary evaluation episodes. The additional 1,440 T-Drive rows with `split=train` evaluate the same final policies on the training vehicle partition. These are diagnostic evaluations, not training episodes or additional independent training runs. They are analyzed separately from held-out results.

The 20 evaluation scenario seeds are shared across policies and training seeds within each mobility, scale, and split. The analysis verifies this pairing, the complete group grid, 20 distinct scenario IDs per group, finite metrics, unique episode identities, and `checkpoint_episode=1000`. It does not treat the 60 evaluation episodes of a method as 60 independent training replicates.

`scale` changes the agent count together with the recorded deployment population, area, and infrastructure configuration. It should not be interpreted as varying only agent count while holding every environmental factor constant.

## CSV fields

Required identity fields are `mobility`, `scale`, `split`, `policy`, `seed`, `evaluation_seed`, and `checkpoint_episode`. `seed` is the training seed; `evaluation_seed` identifies the evaluation scenario. The policy identifiers map to these manuscript names:

| Identifier | Manuscript name |
|---|---|
| `maddpg_srdg_v2` | TwinGraph-MADDPG |
| `maddpg` | MADDPG |
| `matd3` | MATD3 |
| `mappo` | MAPPO |
| `jcce_gat_maddpg` | GAT-MADDPG |
| `jcce_centralized_ddqn` | Centralized DDQN |

The ten metric columns retain the recorded episode summaries:

| Column | Meaning | Favorable direction |
|---|---|---|
| `reward` | Global agent's episodic return | Higher |
| `network_reward` | Sum of episodic returns over all agents | Higher |
| `latency` | Global-cache average lookup cost, using the legacy field name `avg_lookup_latency` | Lower |
| `communication` | Global-cache communication-overhead proxy | Lower |
| `energy` | Global-cache average energy proxy | Lower |
| `valid_hit` | Global-cache valid-hit rate | Higher |
| `fetches` | Global-cache cross-shard fetch count during the episode | Lower |
| `stale` | Recorded global-cache stale-entry audit-event count | Lower |
| `revoked` | Recorded global-cache revoked-entry audit-event count | Lower |
| `requests` | Global-cache request count during the episode | Context only |

Lookup, communication, and energy values are simulator costs or proxies, not measured physical latency, bytes, or joules. `reward` and `network_reward` are distinct quantities. Stale and revoked events are cache audit counters; zero counts do not establish general cryptographic security or a bound on invalid-payload delivery.

## Aggregation and uncertainty

For each policy and training seed, the script first takes the arithmetic mean of the 20 evaluation episode values. All subsequent uncertainty calculations use the **three training-seed means** as their observations.

For seed means `x_1`, `x_2`, and `x_3`, the summary is their arithmetic mean and sample standard deviation (denominator `n - 1`, with `n = 3`). The nominal 95% Student-t interval is:

```text
mean ± 4.302652729696142 × sample_SD / sqrt(3)
```

The interval is calculated without clipping to metric bounds. It is a nominal parametric interval with two degrees of freedom, not a confidence band over training episodes. With only three trained seeds, normality is difficult to assess and uncertainty estimates are imprecise.

Each contrast pairs TwinGraph-MADDPG and a baseline by training seed within the same deployment and split. For each seed the difference is `TwinGraph - baseline`. The contrast mean, sample SD, and Student-t interval are computed from these three differences. The two-sided sign-flip p-value enumerates all `2^3 = 8` sign assignments and counts assignments with absolute mean difference at least as large as observed, using a numerical comparison tolerance of `1e-12`. This enumeration relies on the null sign-exchangeability/symmetry assumption. Its smallest attainable two-sided p-value with three nonzero paired differences is 0.25; it cannot establish significance at 0.05 in this design.

The primary lookup-cost comparisons use Holm correction in **separate families of 20 contrasts** (five baselines times four scales): PPP primary evaluation, T-Drive held-out evaluation, and T-Drive training-partition diagnostic evaluation. The diagnostic family is distinct from the two primary families. For sorted p-values, the adjusted value at rank `i` is the running maximum of `min(1, (20 - i) × p_i)`, using zero-based indexing. Other metric rows have empty Holm fields and remain exploratory; the script does not apply a single correction across all 600 rows. The 40 primary lookup contrasts in the released data have Holm-adjusted p-values of 1.00.

`relative_improvement_pct` is `(baseline - TwinGraph) / baseline × 100` for lower-is-better metrics and `(TwinGraph - baseline) / baseline × 100` for higher-is-better metrics. Positive values indicate the favorable direction when the baseline denominator is positive. The field is empty for request count and when the baseline mean is zero. A signed reward denominator can make a percentage counterintuitive, so the reported raw means and paired differences should also be examined. A favorable mean or percentage is descriptive and does not imply statistical significance.

## Training-clock limitation

The four original policies (TwinGraph-MADDPG, MADDPG, MATD3, and MAPPO) at all PPP scales and the three-agent T-Drive setting were trained with wall-clock cache ages. GAT-MADDPG and centralized DDQN at these settings were trained with logical time. At the five-, seven-, and nine-agent T-Drive settings, all six policies were trained from scratch with the common logical clock. Every released final-policy evaluation uses the same logical-clock rule. Thus common evaluation controls the evaluation clock, but does not remove the training-clock difference in the historical cohorts. The T-Drive 5/7/9 experiments provide the fully matched training-clock comparison.

This public analysis is a transparent, limited artifact: it enables independent recalculation of the released tables and inspection of experiment settings. Full end-to-end reproduction would additionally require the withheld environment and learner implementations, model checkpoints, and applicable mobility-data access.
