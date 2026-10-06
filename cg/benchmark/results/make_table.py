"""Per-pair fpocket results on the Schrodinger set, next to the rankings in si.csv."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SI = Path("~/Dropbox/PocketFinding/SchrodingerSet/si.csv").expanduser()
RUNS = [("AA default", "aa_default"), ("AA tuned", "aa_tuned"), ("Martini 2", "martini2_tuned"),
        ("Martini 3", "martini3_tuned"), ("SIRAH", "sirah_tuned")]  # fmt: skip
OTHERS = ["SiteMap", "MxMD", "SiteMap+MxMD", "P2Rank", "Boltz-1"]

ranks = {}
for label, key in RUNS:
    for r in json.loads((HERE / f"schrodinger_{key}.json").read_text())["results"]:
        ranks[(label, r["kind"], r["case"])] = (r["hit_rank"], r["npockets"])
rows = list(csv.reader(open(SI, encoding="utf-8-sig")))
other = {f"{r[0].strip()}_{r[1].strip()}": r[8:13] for r in rows[2:] if r and r[0].strip()}
cases = sorted({c for _, _, c in ranks})


def cell(label, kind, case):
    rank, n = ranks[(label, kind, case)]
    return f"{rank}" if rank else f"NF/{n}"


header = ["pair", "ligand"] + [f"apo {lab}" for lab, _ in RUNS] + [f"holo {lab}" for lab, _ in RUNS] + OTHERS
lig = {f"{r[0].strip()}_{r[1].strip()}": r[2].strip() for r in rows[2:] if r and r[0].strip()}
table = [[c, lig[c]] + [cell(lab, "apo", c) for lab, _ in RUNS] + [cell(lab, "holo", c) for lab, _ in RUNS]
         + [x.strip() for x in other[c]] for c in cases]  # fmt: skip
with open(HERE / "schrodinger_ranks.csv", "w", newline="") as f:
    csv.writer(f).writerows([header, *table])


def top(label, kind, k):
    hits = [ranks[(label, kind, c)][0] for c in cases]
    return sum(1 for h in hits if h and h <= k) / len(cases)


def top_other(i, k):
    vals = [other[c][i].strip() for c in cases]
    return sum(1 for v in vals if v.isdigit() and int(v) <= k) / len(cases)


md = ["| pair | lig | " + " | ".join(lab for lab, _ in RUNS) + " | " + " | ".join(lab for lab, _ in RUNS)
      + " | " + " | ".join(OTHERS) + " |",
      "|---|---|" + "---|" * (2 * len(RUNS) + len(OTHERS))]  # fmt: skip
md += ["| " + " | ".join(row) + " |" for row in table]
for k in (1, 3, 5):
    md.append(f"| **Top-{k}** | | " + " | ".join(f"{top(lab, 'apo', k):.2f}" for lab, _ in RUNS) + " | "
              + " | ".join(f"{top(lab, 'holo', k):.2f}" for lab, _ in RUNS) + " | "
              + " | ".join(f"{top_other(i, k):.2f}" for i in range(len(OTHERS))) + " |")  # fmt: skip
(HERE / "schrodinger_ranks.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
