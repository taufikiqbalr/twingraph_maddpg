#!/usr/bin/env python3
"""Recalculate the JCCE tables from the released episode-level evaluation data.

Uses the Python standard library only. This script analyzes recorded evaluations;
it does not train policies, execute the simulator, or recover withheld code.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
POLICIES = (
    "maddpg_srdg_v2", "maddpg", "matd3", "mappo",
    "jcce_gat_maddpg", "jcce_centralized_ddqn",
)
LABELS = dict(zip(POLICIES, (
    "TwinGraph-MADDPG", "MADDPG", "MATD3", "MAPPO", "GAT-MADDPG",
    "Centralized DDQN",
)))
SEEDS = (11, 17, 23)
SCALES = (3, 5, 7, 9)
METRICS = (
    "reward", "network_reward", "latency", "communication", "energy",
    "valid_hit", "fetches", "stale", "revoked", "requests",
)
ID_COLUMNS = (
    "mobility", "scale", "split", "policy", "seed", "evaluation_seed",
    "checkpoint_episode",
)
T_CRITICAL_DF2 = 4.302652729696142  # Student-t 0.975 quantile, 3 seeds - 1 df
LOWER_IS_BETTER = {"latency", "communication", "energy", "fetches", "stale", "revoked"}
HIGHER_IS_BETTER = {"reward", "network_reward", "valid_hit"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evaluation_kind(mobility, split):
    if mobility == "ppp":
        return "synthetic_fresh_scenarios"
    return "held_out_vehicles" if split == "eval" else "training_vehicle_diagnostic"


def expected_groups():
    return {
        (mobility, scale, split, policy, seed)
        for mobility in ("ppp", "tdrive")
        for scale in SCALES
        for split in (("eval", "train") if mobility == "tdrive" else ("eval",))
        for policy in POLICIES
        for seed in SEEDS
    }


def read_and_validate(path):
    """Reject missing, duplicate, malformed, or unpaired evaluation records."""
    groups = defaultdict(list)
    identities = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(ID_COLUMNS + METRICS)
        require(reader.fieldnames is not None, "Input CSV has no header")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate CSV column names")
        require(required.issubset(reader.fieldnames),
                "Missing CSV columns: " + ", ".join(sorted(required - set(reader.fieldnames))))
        for line, raw in enumerate(reader, start=2):
            require(None not in raw and all(raw.get(k) not in (None, "") for k in required),
                    f"Malformed or empty required field at CSV line {line}")
            try:
                row = {k: raw[k] for k in ("mobility", "split", "policy")}
                row.update({k: int(raw[k]) for k in
                            ("scale", "seed", "evaluation_seed", "checkpoint_episode")})
                row.update({metric: float(raw[metric]) for metric in METRICS})
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid numeric value at CSV line {line}") from error
            require(row["checkpoint_episode"] == 1000,
                    f"Expected the final 1,000-episode checkpoint at CSV line {line}")
            require(all(math.isfinite(row[m]) for m in METRICS),
                    f"Nonfinite metric at CSV line {line}")
            key = tuple(row[k] for k in ("mobility", "scale", "split", "policy", "seed"))
            identity = key + (row["evaluation_seed"],)
            require(identity not in identities, f"Duplicate evaluation identity at CSV line {line}: {identity}")
            identities.add(identity)
            groups[key].append(row)

    expected = expected_groups()
    missing, extra = expected - groups.keys(), groups.keys() - expected
    require(not missing and not extra,
            f"Incomplete or unexpected evaluation grid: {len(missing)} missing, {len(extra)} unexpected groups")
    require(len(identities) == 4320, f"Expected 4,320 evaluation records, found {len(identities)}")
    scenarios = {}
    for key, rows in groups.items():
        require(len(rows) == 20, f"Expected 20 evaluation episodes per trained seed: {key}")
        scenario_key = key[:3]
        ids = frozenset(row["evaluation_seed"] for row in rows)
        if scenario_key in scenarios:
            require(ids == scenarios[scenario_key], f"Evaluation scenarios differ across policies/seeds: {key}")
        else:
            scenarios[scenario_key] = ids
    return groups


def interval(values):
    """Sample SD and nominal Student-t interval across three trained seeds."""
    require(len(values) == 3, "The released analysis requires exactly three training seeds")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T_CRITICAL_DF2 * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def summarize(groups):
    seed_means = []
    for (mobility, scale, split, policy, seed), rows in sorted(groups.items()):
        seed_means.append(dict(
            mobility=mobility, scale=scale, split=split,
            evaluation_kind=evaluation_kind(mobility, split),
            policy=policy, policy_label=LABELS[policy], seed=seed,
            checkpoint_episode=1000, evaluation_clock="logical_one_second_per_step",
            n_evaluation_episodes=len(rows),
            **{metric: statistics.mean(r[metric] for r in rows) for metric in METRICS},
        ))
    index = {
        (r["mobility"], r["scale"], r["split"], r["policy"], r["seed"]): r
        for r in seed_means
    }
    summaries, contrasts = [], []
    for mobility, scale in itertools.product(("ppp", "tdrive"), SCALES):
        for split in (("eval", "train") if mobility == "tdrive" else ("eval",)):
            for policy, metric in itertools.product(POLICIES, METRICS):
                values = [index[mobility, scale, split, policy, seed][metric] for seed in SEEDS]
                mean, sd, low, high = interval(values)
                summaries.append(dict(
                    mobility=mobility, scale=scale, split=split,
                    evaluation_kind=evaluation_kind(mobility, split),
                    policy=policy, policy_label=LABELS[policy], metric=metric,
                    n_training_seeds=3, n_evaluation_episodes_per_seed=20,
                    mean=mean, sd=sd, ci95_low=low, ci95_high=high,
                ))
                if policy == "maddpg_srdg_v2":
                    continue
                twin_values = [index[mobility, scale, split, "maddpg_srdg_v2", seed][metric]
                               for seed in SEEDS]
                differences = [twin - baseline for twin, baseline in zip(twin_values, values)]
                dm, ds, dl, dh = interval(differences)
                permuted = [
                    abs(statistics.mean(d * sign for d, sign in zip(differences, signs)))
                    for signs in itertools.product((-1, 1), repeat=3)
                ]
                p_value = sum(value >= abs(dm) - 1e-12 for value in permuted) / 8
                twin_mean = statistics.mean(twin_values)
                if mean and metric in LOWER_IS_BETTER:
                    improvement = (mean - twin_mean) / mean * 100
                elif mean and metric in HIGHER_IS_BETTER:
                    improvement = (twin_mean - mean) / mean * 100
                else:
                    improvement = None
                contrasts.append(dict(
                    mobility=mobility, scale=scale, split=split,
                    baseline=policy, baseline_label=LABELS[policy], metric=metric, n_pairs=3,
                    difference_twingraph_minus_baseline=dm, sd=ds,
                    ci95_low=dl, ci95_high=dh, exact_two_sided_p=p_value,
                    holm_lookup_cost_p=None, holm_family=None,
                    relative_improvement_pct=improvement,
                    twingraph_mean=twin_mean, baseline_mean=mean,
                ))

    # The primary cost family is 5 baselines x 4 scales per mobility/split.
    # Other metrics are exploratory and are not adjusted by this family.
    for mobility, split in (("ppp", "eval"), ("tdrive", "eval"), ("tdrive", "train")):
        family = sorted(
            (r for r in contrasts if r["mobility"] == mobility and
             r["split"] == split and r["metric"] == "latency"),
            key=lambda r: r["exact_two_sided_p"],
        )
        require(len(family) == 20, "A Holm lookup-cost family must contain 20 contrasts")
        running = 0
        for i, row in enumerate(family):
            running = max(running, min(1, (20 - i) * row["exact_two_sided_p"]))
            row["holm_lookup_cost_p"] = running
            row["holm_family"] = f"{mobility}_{split}_lookup_cost_20_contrasts"
    require((len(seed_means), len(summaries), len(contrasts)) == (216, 720, 600),
            "Unexpected output dimensions")
    return seed_means, summaries, contrasts


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=PACKAGE_ROOT / "data" / "evaluation_episodes.csv",
                        help="Episode CSV (default: this release's data/evaluation_episodes.csv)")
    parser.add_argument("--output", type=Path,
                        default=PACKAGE_ROOT / "outputs" / "tables",
                        help="Output directory (default: this release's outputs/tables)")
    args = parser.parse_args()
    try:
        names = ("seed_means.csv", "metric_summary.csv", "contrasts.csv")
        require(all((args.output / name).resolve() != args.data.resolve() for name in names),
                "Output would overwrite the input data")
        groups = read_and_validate(args.data)
        tables = summarize(groups)
        args.output.mkdir(parents=True, exist_ok=True)
        for name, rows in zip(names, tables):
            write_csv(args.output / name, rows)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    print(json.dumps({
        "status": "PASS", "evaluation_episodes": 4320,
        "primary_evaluation_episodes": 2880,
        "tdrive_training_partition_diagnostic_episodes": 1440,
        "seed_mean_rows": 216, "metric_summary_rows": 720,
        "contrast_rows": 600, "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
