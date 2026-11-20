#!/usr/bin/env python3
"""Verify the release manifest and complete recorded-data experiment grid."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('maddpg_srdg_v2','maddpg','matd3','mappo','jcce_gat_maddpg','jcce_centralized_ddqn')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT,help='Extracted release directory')
    args=parser.parse_args();root=args.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for record in manifest['files']:
        path=(root/record['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or invalid release path: '+record['path'])
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=record['sha256'] or path.stat().st_size!=record['bytes']:
            raise ValueError('Release file differs from manifest: '+record['path'])
    groups=defaultdict(list)
    with gzip.open(root/'data/training_episodes.csv.gz','rt',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(row['mobility'],int(row['scale']),row['policy'],int(row['seed']))
            groups[key].append(int(row['episode']))
            if not all(math.isfinite(float(row[m])) for m in ('reward','latency','fetches')):
                raise ValueError('Non-finite training metric')
    expected=set(itertools.product(('ppp','tdrive'),(3,5,7,9),POLICIES,(11,17,23)))
    if set(groups)!=expected or any(v!=list(range(1,1001)) for v in groups.values()):
        raise ValueError('Training records do not cover all 144 complete runs')
    # The analysis script contains the complete evaluation pairing validation.
    from analyze_results import read_and_validate
    read_and_validate(root/'data/evaluation_episodes.csv')
    print(json.dumps(dict(status='PASS',verified_files=len(manifest['files']),
        training_runs=len(groups),training_episodes=144000,evaluation_episodes=4320,
        scope='Recorded data and configuration integrity; no training or policy inference'),indent=2))


if __name__=='__main__':main()
