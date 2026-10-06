"""fpocket on every Schrodinger apo structure, per setting, with an all-atom PyMOL view per pair.

For each apo/holo pair of si.csv (not marked EXCLUDE, both files present)
and each setting, ``cgpocket.py run`` finds pockets in the apo structure,
carries the holo ligand (resname LIG) onto the apo by boonza.superpose of the
holo protein on the apo protein, scores every pocket against it (PPc, MOc), and
writes ``<out>/fpocket_<setting>/<apo>_<holo>/``:

    view.pml               PyMOL: apo, holo and ligand (MAE, bond orders kept)
    apo.mae holo.mae ligand.mae pockets.pqr
    pockets_vs_holo.csv    every pocket: distances to the ligand, PPc, MOc
    fpocket_info.txt       fpocket's descriptors and scores
    fpocket/               fpocket's own run (its input PDB and output)

and ``fpocket_<setting>/summary.csv`` / ``summary.md`` (Top-1/3/5/10).

Settings:  default -- all-atom fpocket, its own flags and score;
optimized -- all-atom, fpocket's flags with the score refitted on fpocket's
training set (results/aa_refit_reference.json); martini2, martini3, sirah --
the presets of ../presets.json.

    python run_views.py --data ~/Dropbox/PocketFinding/SchrodingerSet \\
        --settings default optimized martini2 martini3
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import prepare
import score_fit as F

HERE = Path(__file__).resolve().parent
CGPOCKET = HERE.parent / "cgpocket.py"
SETTINGS = ("default", "optimized", "martini2", "martini3", "sirah")


def setting_args(setting: str) -> list[str]:
    """--model and extra fpocket flags for a setting."""
    if setting == "default":
        return ["--model", "aa"]
    if setting == "optimized":
        coeffs = json.loads((HERE / "results" / "aa_refit_reference.json").read_text())["coefficients"]
        return ["--model", "aa", "--", F.coefficient_flag(coeffs)]
    return ["--model", setting]


def run_pair(job):
    setting, pair, out = job
    where = out / f"fpocket_{setting}" / f"{pair['apo']}_{pair['holo']}"
    where.mkdir(parents=True, exist_ok=True)
    model_args = setting_args(setting)
    extra = model_args[model_args.index("--"):] if "--" in model_args else []
    model_args = model_args[: len(model_args) - len(extra)]
    cmd = [sys.executable, str(CGPOCKET), "run", pair["apo_file"], *model_args,
           "-o", str(where / "fpocket"), "--holo", pair["holo_file"], "--holo-ligand", "resname LIG",
           "--view-dir", str(where)]  # fmt: skip
    cmd += extra
    with open(where / "cgpocket.log", "w") as log:
        done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return setting, pair["case"], where, done.returncode


def first_ranks(table: Path):
    rows = list(csv.reader(open(table)))[1:]
    ppc = next((int(r[0]) for r in rows if r[6] == "True"), None)
    moc = next((int(r[0]) for r in rows if r[7] == "True"), None)
    top_dca = float(rows[0][2]) if rows else float("nan")
    return ppc, moc, len(rows), top_dca


def summarize(folder: Path, pairs) -> str:
    lines = [["pair", "ligand", "pockets", "first_PPc_rank", "first_MOc_rank", "top_pocket_center_to_ligand"]]
    ranks = {"PPc": [], "MOc": []}
    for p in pairs:
        table = folder / f"{p['apo']}_{p['holo']}" / "pockets_vs_holo.csv"
        if not table.exists():
            lines.append([f"{p['apo']}_{p['holo']}", "LIG", "failed", "", "", ""])
            ranks["PPc"].append(None)
            ranks["MOc"].append(None)
            continue
        ppc, moc, n, dca = first_ranks(table)
        ranks["PPc"].append(ppc)
        ranks["MOc"].append(moc)
        lines.append([f"{p['apo']}_{p['holo']}", "LIG", n, ppc or "", moc or "", f"{dca:.2f}"])
    with open(folder / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(lines)
    n = len(pairs)
    table = {c: [sum(1 for r in v if r is not None and r <= k) / n for k in (1, 3, 5, 10)]
             for c, v in ranks.items()}  # fmt: skip
    md = [f"# {folder.name}: {n} apo structures", "",
          "Fraction of apo structures with a correct pocket among the first k pockets.", "",
          "| criterion | Top-1 | Top-3 | Top-5 | Top-10 |", "|---|---|---|---|---|"]  # fmt: skip
    md += [f"| {c} | " + " | ".join(f"{x:.2f}" for x in v) + " |" for c, v in table.items()]
    md += ["", "PPc: pocket center < 4 Å from a ligand atom.  MOc: > 50% of ligand atoms within",
           "3 Å of the pocket's spheres and > 20% of its spheres within 3 Å of the ligand.",
           "Per pair: summary.csv; per pocket: <pair>/pockets_vs_holo.csv."]  # fmt: skip
    (folder / "summary.md").write_text("\n".join(md) + "\n")
    return "  ".join(f"{c} " + " / ".join(f"{x:.2f}" for x in v) for c, v in table.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="default: --data")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS[:4]), choices=SETTINGS)
    ap.add_argument("-j", "--jobs", type=int, default=4)
    args = ap.parse_args()
    out = args.out or args.data
    pairs = prepare.schrodinger_pairs(args.data)
    jobs = [(s, p, out) for s in args.settings for p in pairs]
    print(f"{len(pairs)} pairs x {len(args.settings)} settings", flush=True)
    failed = []
    with ProcessPoolExecutor(args.jobs) as pool:
        for setting, case, where, code in pool.map(run_pair, jobs):
            if code:
                failed.append((setting, case, where / "cgpocket.log"))
                print(f"FAILED {setting} {case}: see {where / 'cgpocket.log'}", flush=True)
    for s in args.settings:
        print(f"{s:10s} {summarize(out / f'fpocket_{s}', pairs)}")
    if failed:
        sys.exit(f"{len(failed)} runs failed")


if __name__ == "__main__":
    main()
