"""Print natural-shade tables (L quantiles + (a, b) at shadow/mid/highlight) from results/review/hair_stats.csv.

Run scripts/hair_stats.py first. The output is pasted into src/vto_spike/shades.py with this script named as source.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = [("black", 0, 12), ("dark_brown", 12, 22), ("brown", 22, 35), ("light_brown", 35, 50), ("blonde", 50, 101)]

rows = []
with open(ROOT / "results" / "review" / "hair_stats.csv") as f:
    for r in csv.DictReader(f):
        rows.append({k: (float(v) if k != "stem" else v) for k, v in r.items()})

for name, lo, hi in FAMILIES:
    g = [r for r in rows if lo <= r["L_p50"] < hi]
    m = lambda k: float(np.nanmean([r[k] for r in g]))  # noqa: E731
    print(f'    "{name}": Shade(  # n={len(g)}')
    print(f"        L=({m('L_p2'):.1f}, {m('L_p10'):.1f}, {m('L_p50'):.1f}, {m('L_p90'):.1f}, {m('L_p98'):.1f}),")
    print(f"        ab_shadow=({m('a_shadow'):.1f}, {m('b_shadow'):.1f}), ab_mid=({m('a_mid'):.1f}, {m('b_mid'):.1f}),"
          f" ab_high=({m('a_high'):.1f}, {m('b_high'):.1f}),")
    print("    ),")
