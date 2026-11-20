# Hyperparameters and architecture

These records describe the 144 policies used in the revised JCCE comparisons. All six methods use 1,000 training episodes, 80 simulator steps per episode, and three independent training seeds (11, 17, 23) in each of the eight mobility/deployment settings. No 100-episode comparison is included.

The JSON files are selected declarative records, not trainer entry points. This public package supports examination of the reported settings and reproduction of analyses from released metrics. The full training pipeline, model implementations and complete internal configuration are not distributed; the files alone cannot retrain the policies.

## Learning settings

| Method | Actor / Q learning rate | Critic learning rate | Hidden width | Batch / minibatch | Discount | Target tau | Update schedule |
|---|---:|---:|---:|---:|---:|---:|---|
| TwinGraph-MADDPG | 0.00021 | 0.00044 | 160 | 64 | 0.97 | 0.016 | 4 learner updates per environment step; actor delay 2 |
| MADDPG | 0.00030 | 0.00080 | 96 | 48 | 0.97 | 0.020 | 2 learner updates per environment step; actor delay 1 |
| MATD3 | 0.00030 | 0.00080 | 96 | 48 | 0.97 | 0.020 | 2 learner updates per environment step; actor delay 2 |
| MAPPO | 0.00030 | 0.00070 | 96 | 16 | 0.96 | Not applicable | 80-step rollout; 12 optimization epochs |
| GAT-MADDPG | 0.00030 | 0.00080 | 96 | 48 | 0.97 | 0.020 | 2 learner updates per environment step; actor delay 1 |
| Centralized DDQN | 0.00030 | Not applicable | 160 | 64 | 0.97 | 0.020 | 2 learner updates per environment step |

For off-policy methods, the target update is `target <- (1 - tau) * target + tau * online`. MADDPG-family target networks update on actor-update cycles; DDQN updates its target after each learner update. Replay capacity is 12,000 transitions. MADDPG-family replay stores agent transitions; centralized DDQN stores joint environment transitions. Updates begin after a sufficient replay batch is available. Thus equal episode budgets do not imply equal gradient-update counts; `run_manifest.csv` records the realized learner cycles.

TwinGraph uses AdamW with actor/critic weight decay 0.00001/0.00005. The other methods use Adam with zero weight decay. Adam-family beta parameters are (0.9, 0.999), and optimizer epsilon is 0.00000001. Off-policy gradient norm clipping is 0.8; MAPPO uses 0.75. MAPPO uses GAE lambda 0.92, clipping ratio 0.20, entropy coefficient 0.006 and value coefficient 0.7. Further selected parameters are in the six model JSON files.

Exploration is scale-dependent for TwinGraph. In both PPP and T-Drive, TwinGraph uses initial epsilon 0.25, per-episode decay 0.995 and minimum epsilon 0.03 at 3/5 agents; at 7/9 agents the corresponding values are 0.28, 0.998 and 0.05. MADDPG, MATD3, GAT-MADDPG and centralized DDQN use 0.25, 0.995 and 0.03 at every scale. The update is `epsilon <- max(minimum, epsilon * decay)` after each training episode. These are exploration parameters, distinct from the optimizer epsilon above. MAPPO samples its categorical policy during training and does not use this epsilon-greedy schedule. Every applicable condition in `configs/conditions.json` includes its resolved initial epsilon, decay and floor; the TwinGraph model file explicitly lists both scale schedules. The final policies are evaluated greedily.

## Architecture dimensions

All actor-critic methods share a local actor across agents. The actor applies a two-hidden-layer ReLU MLP independently to each candidate, using the agent summary and the candidate features. Hidden widths are 160/160 for TwinGraph and 96/96 for MADDPG, MATD3, MAPPO and GAT-MADDPG. The output is seven action logits per candidate. Each agent has at most eight candidate slots. The candidate feature width is 55, comprising 31 base features and 24 temporal features from a four-step history.

| Component | Architecture |
|---|---|
| TwinGraph critics | Two independent graph critics, hidden width 160; two candidate attention blocks before and two after graph propagation; two agent attention blocks; four heads per block; attention feed-forward width 320; two communication rounds and four roles |
| MADDPG critic | Centralized action critic, two ReLU hidden layers of width 96 and one scalar Q output |
| MATD3 critics | Two independent centralized action critics, each with two ReLU hidden layers of width 96 and one scalar Q output |
| MAPPO critic | Centralized state-value network, two ReLU hidden layers of width 96 and one scalar output |
| GAT-MADDPG critic | Two graph-attention layers for each of the candidate and agent graphs, four heads per layer, total width 96 (24 per head); one 96-unit ReLU Q-readout hidden layer |
| Centralized DDQN | Two 160-unit ReLU trunk layers; one scalar value head and one advantage projection covering eight candidate branches per agent, seven actions per branch |

TwinGraph has a branched architecture. The listed depths refer to individual modules and should not be read as a single sequential total-layer count. The 160/160 candidate and agent encoders and the attention depths document network capacity without distributing the private graph, gating, auxiliary-loss or simulator implementations.

| Agents | Summary width per agent | Actor observation width | DDQN input width | DDQN action branches |
|---:|---:|---:|---:|---:|
| 3 | 26 | 466 | 1398 | 24 |
| 5 | 28 | 468 | 2340 | 40 |
| 7 | 30 | 470 | 3290 | 56 |
| 9 | 32 | 472 | 4248 | 72 |

Actor observation width is `summary width + 8 * 55`. DDQN concatenates all agents' observations. `conditions.json` reports the dimensions and resolved learning settings for all 48 method/mobility/scale conditions.

## Recorded training duration

The following ranges span the three independent training seeds and are measured hours per trained policy. Per-run seconds, hours, realized updates, devices, timing definitions and retained software metadata are in `configs/run_manifest.csv`. Unrecorded software fields are blank instead of inferred from the current machine.

Archived CUDA runs report the sum of training-episode durations and exclude checkpoint/log I/O outside the episode loop. Later runs report active worker/loop time, including training I/O. All exclude evaluation and startup. Work was run concurrently, and hardware, timing scopes and implementation execution differ. These values describe the recorded cost of the experiments; they are not controlled speed or GPU-acceleration comparisons.

| Mobility | Agents | Method | Training hours, min–max | Device |
|---|---:|---|---:|---|
| PPP | 3 | TwinGraph-MADDPG | 19.213–27.279 | CUDA |
| PPP | 3 | MADDPG | 0.432–0.761 | CUDA |
| PPP | 3 | MATD3 | 0.405–0.777 | CUDA |
| PPP | 3 | MAPPO | 0.390–0.392 | CUDA |
| PPP | 3 | GAT-MADDPG | 0.709–0.712 | CPU |
| PPP | 3 | Centralized DDQN | 0.423–0.432 | CPU |
| PPP | 5 | TwinGraph-MADDPG | 39.472–40.361 | CUDA |
| PPP | 5 | MADDPG | 1.069–1.189 | CUDA |
| PPP | 5 | MATD3 | 1.073–1.188 | CUDA |
| PPP | 5 | MAPPO | 0.782–0.793 | CUDA |
| PPP | 5 | GAT-MADDPG | 2.328–2.336 | CUDA |
| PPP | 5 | Centralized DDQN | 1.069–1.082 | CUDA |
| PPP | 7 | TwinGraph-MADDPG | 32.390–36.000 | CUDA |
| PPP | 7 | MADDPG | 1.508–1.617 | CUDA |
| PPP | 7 | MATD3 | 1.511–1.614 | CUDA |
| PPP | 7 | MAPPO | 1.208–1.225 | CUDA |
| PPP | 7 | GAT-MADDPG | 2.752–2.757 | CUDA |
| PPP | 7 | Centralized DDQN | 1.465–1.477 | CUDA |
| PPP | 9 | TwinGraph-MADDPG | 31.751–32.291 | CUDA |
| PPP | 9 | MADDPG | 2.022–2.057 | CUDA |
| PPP | 9 | MATD3 | 2.032–2.045 | CUDA |
| PPP | 9 | MAPPO | 1.800–1.813 | CUDA |
| PPP | 9 | GAT-MADDPG | 3.169–3.183 | CUDA |
| PPP | 9 | Centralized DDQN | 1.949–1.967 | CUDA |
| T-Drive | 3 | TwinGraph-MADDPG | 25.454–30.458 | CUDA |
| T-Drive | 3 | MADDPG | 0.884–0.932 | CUDA |
| T-Drive | 3 | MATD3 | 0.895–0.955 | CUDA |
| T-Drive | 3 | MAPPO | 0.550–0.551 | CUDA |
| T-Drive | 3 | GAT-MADDPG | 0.943–0.950 | CPU |
| T-Drive | 3 | Centralized DDQN | 0.637–0.648 | CPU |
| T-Drive | 5 | TwinGraph-MADDPG | 49.520–50.019 | CPU |
| T-Drive | 5 | MADDPG | 1.232–1.270 | CPU |
| T-Drive | 5 | MATD3 | 1.264–1.293 | CPU |
| T-Drive | 5 | MAPPO | 1.043–1.052 | CPU |
| T-Drive | 5 | GAT-MADDPG | 1.683–1.716 | CPU |
| T-Drive | 5 | Centralized DDQN | 1.322–1.357 | CPU |
| T-Drive | 7 | TwinGraph-MADDPG | 56.882–57.157 | CPU |
| T-Drive | 7 | MADDPG | 1.966–1.981 | CPU |
| T-Drive | 7 | MATD3 | 1.934–1.954 | CPU |
| T-Drive | 7 | MAPPO | 1.721–1.734 | CPU |
| T-Drive | 7 | GAT-MADDPG | 2.361–2.394 | CPU |
| T-Drive | 7 | Centralized DDQN | 2.134–2.150 | CPU |
| T-Drive | 9 | TwinGraph-MADDPG | 63.931–64.127 | CPU |
| T-Drive | 9 | MADDPG | 2.816–2.831 | CPU |
| T-Drive | 9 | MATD3 | 2.799–2.822 | CPU |
| T-Drive | 9 | MAPPO | 2.649–2.684 | CPU |
| T-Drive | 9 | GAT-MADDPG | 3.299–3.340 | CPU |
| T-Drive | 9 | Centralized DDQN | 3.079–3.099 | CPU |

The recorded GPU is an NVIDIA GeForce RTX 5090. The CPU workstation is an AMD Ryzen Threadripper 9960X. The larger T-Drive runs used single-thread CPU workers; the additional PPP baseline campaign used CUDA workers. Configurations retaining runtime metadata report Python 3.12.13, PyTorch 2.10.0+cu130, NumPy 2.4.4 and CUDA 13.0, with deterministic algorithms disabled. Software details not retained in older configurations are left unspecified in the run manifest.

## Interpretation and release boundary

These baselines are the concrete implementations used in this caching simulator, not claims of universal default settings for their algorithm families. Replay organization, auxiliary terms and optimization schedules differ. Several operational terms and the full model construction remain in the restricted implementation. This release therefore addresses disclosure of the listed hyperparameters and reproducibility of the published numerical analysis, but does not claim independent end-to-end retraining.
