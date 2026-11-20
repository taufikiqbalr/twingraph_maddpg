# Training and evaluation protocol

## Study grid

The comparison has six methods × two mobility settings × four deployment scales × three training seeds: 144 policies in 48 conditions. Every policy completed 1,000 training episodes with 80 simulator steps each (80,000 environment interactions per policy). The total is 144,000 training episodes and 11,520,000 environment interactions. The training seeds are 11, 17 and 23. The final fixed-budget checkpoint was selected without using evaluation performance to choose a checkpoint.

The configurations are descriptive experiment records. The public repository does not contain an executable simulator/trainer, model weights or the complete internal controller configuration. Consequently these records permit inspection of the study design and support result-analysis reproduction; they do not provide an independent training reproduction.

## Deployment settings

| Cache agents | Global / regional agents | Base stations | Devices | Plane extent (simulator coordinates) |
|---:|---:|---:|---:|---|
| 3 | 1 / 2 | 12 | 60 | 120 × 120 |
| 5 | 1 / 4 | 20 | 100 | 160 × 160 |
| 7 | 1 / 6 | 30 | 150 | 190 × 190 |
| 9 | 1 / 8 | 40 | 200 | 220 × 220 |

There are five devices per base station, eight candidate slots per cache agent and seven actions per candidate. The nominal base-station intensity is 0.0008 per square simulator unit; the minimum and maximum station counts are both fixed to the table values. Agent count, infrastructure, resident population and area change jointly. These are deployment-scale comparisons, not isolated agent-count interventions.

PPP uses synthetic mobility. T-Drive uses recorded vehicle trajectories with a minimum of 100 points per usable vehicle file and relative anchoring to the simulated area. Training and primary evaluation use disjoint vehicle files. The separate T-Drive training-partition evaluation is a diagnostic and must not be pooled with held-out results. Raw T-Drive trajectories and the full private dataset inventory are not redistributed here.

## Retained training-clock differences

| Mobility / scale | TwinGraph, MADDPG, MATD3, MAPPO | GAT-MADDPG, centralized DDQN |
|---|---|---|
| PPP, all four scales | Archived wall-clock cache ages | Logical-clock cache ages |
| T-Drive, 3 agents | Archived wall-clock cache ages | Logical-clock cache ages |
| T-Drive, 5 / 7 / 9 agents | Logical-clock cache ages | Logical-clock cache ages |

The logical clock advances one second per simulator step. All final policy evaluations use this common logical clock. Common evaluation controls the testing workload but does not erase differences in earlier training. The T-Drive 5/7/9-agent conditions provide the strongest comparison under common learning-clock conditions. The training and evaluation clock for each condition is recorded in `configs/conditions.json`.

## Seeds

In `configs/run_manifest.csv`, each run lists the exact first and last environment-training seed and increment. All runs use 1,000 consecutive seeds.

- Archived four-method runs and three-agent PPP GAT/DDQN use `training_seed + zero_based_episode`.
- Additional PPP baseline runs at 5/7/9 agents use `1000000000 + scenario_index * 100000000 + training_seed * 100000 + zero_based_episode`, with scenario indices 0/1/2 for 5/7/9 agents.
- Logical-clock T-Drive runs use `agents * 100000000 + training_seed * 100000 + zero_based_episode`.

Thus three training-seed labels identify independently initialized policies; they are distinct from environment episode seeds. The archived sliding seed windows overlap heavily across training seeds, which should be retained when interpreting replication.

## Final-policy evaluation

Every trained policy is evaluated greedily with learning disabled on 20 common scenarios for its deployment. PPP evaluation uses fresh synthetic scenarios; the primary T-Drive evaluation uses held-out vehicle files. This gives 2,880 primary evaluation episodes. A separate 20-scenario training-partition diagnostic for each T-Drive policy adds 1,440 episodes, yielding 4,320 recorded evaluation episodes overall.

Evaluation seed ranges are 7,500,000–7,500,019 for three-agent PPP; `1900000000 + agents * 1000000 + [0..19]` for PPP at 5/7/9 agents; and `2000000000 + agents * 1000000 + [0..19]` for each T-Drive deployment. The held-out and training-partition T-Drive evaluations use the same seed numbers with different vehicle partitions. `configs/protocol.json` provides every evaluation seed explicitly.

Metrics are first averaged over the 20 scenarios for each trained policy. The reported mean and sample SD then use the three independent training-seed means. Evaluation episodes are repeated measures; they are not 60 independent training replications. Primary paired lookup-cost comparisons use three seed pairs, nominal Student-t intervals, exact two-sided sign-flip tests and separate Holm families of 20 contrasts for PPP and held-out T-Drive.

## Training curves and timing

Training curves apply a trailing 20-episode arithmetic mean separately to each seed, using the available prefix during the first 19 episodes. The line and band show the mean and sample SD across three smoothed seed series. These plots describe exploratory training, not frozen-policy evaluation or additional statistical replications.

Duration ranges and per-run timing scopes are in `docs/HYPERPARAMETERS.md`, `configs/training_duration_summary.csv` and `configs/run_manifest.csv`. Concurrent execution, hardware differences and different update schedules prevent interpreting the recorded durations as controlled algorithmic speed comparisons.

## Public record boundaries

The public configurations were selected from retained experiment metadata and checked against all 144 training logs. No current private training implementation is exposed through these records. Standard hyperparameters and module dimensions are reported explicitly; undistributed internal settings are not replaced with guessed values or misleading runnable stubs. To reproduce training end to end, the simulator, feature/graph construction, full controllers, exact dataset partition implementation and complete operational settings would also be required.
