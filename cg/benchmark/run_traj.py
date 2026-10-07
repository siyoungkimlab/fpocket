"""mdpocket on the coarse-grained probe trajectories of a run directory, one entry per apo/holo pair.

For each ``<root>/<model>/<apo>/<probes>/sim_000/md_solute/{solvated.dms,trajectory.dcd}``
(model martini2, martini3 or sirah) and each apo/holo pair of si.csv not
marked EXCLUDE, ``cgpocket.py traj`` keeps the protein beads (never the
chain-LIG probes), fits every frame on the first by the backbone beads, runs
mdpocket with the model's preset and writes ``<out>/<model>/<apo>_<holo>/``:

    view.pml                   PyMOL: the all-atom apo, the holo ligand, the consensus
                               pockets by quality rank, the frequency surface, the density mesh
    pockets_vs_holo.csv        consensus pockets ranked three ways (cgpocket.py traj --rank):
                               persistence, quality, quality x buriedness; PPc, MOc
    consensus_pockets.csv fpocket_info.txt frames.csv consensus_pockets.pqr
    apo.mae holo_<holo>.mae ligand_<holo>.mae pocket_frequency.dx pocket_density.dx
    ligand_site_frequency_<holo>.csv   per ligand atom: pocket frequency and density within 2 Å
    mdpocket/                  the beads (md.pdb, md.dcd) and mdpocket's own output
    cgpocket.log

and ``<out>/<model>/summary.csv`` / ``summary.md``.  An apo with two holo
structures (2SHP, 2CM2) gives two entries over the same trajectory; an apo
with no valid pair gives none.  si.csv is read only for the pairs.

    python run_traj.py --root ~/Dropbox/PocketFinding/SchrodingerSet/20261005_apo \\
        --data ~/Dropbox/PocketFinding/SchrodingerSet --stride 2
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import prepare

HERE = Path(__file__).resolve().parent
CGPOCKET = HERE.parent / "cgpocket.py"
MODELS = ("martini2", "martini3", "sirah")


def find_runs(root: Path, models) -> list[tuple[str, str, Path]]:
    runs = []
    for model in models:
        for traj in sorted((root / model).glob("*/*/sim_*/md_solute/trajectory.dcd")):
            apo = traj.relative_to(root / model).parts[0]
            if (traj.parent / "solvated.dms").exists():
                runs.append((model, apo, traj.parent))
    return runs


def run_one(job):
    model, apo, md, apo_file, holos, out, stride, cutoff, merge, merge_iso = job
    where = out / model / f"{apo}_{holos[0].name.split('.')[0]}"
    where.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(CGPOCKET), "traj", "--coords", str(md / "solvated.dms"),
           "--traj", str(md / "trajectory.dcd"), "--model", model, "-o", str(where / "mdpocket" / "md"),
           "--run", "--rank", "--apo", str(apo_file), "--view-dir", str(where)]  # fmt: skip
    if stride and stride > 1:
        cmd += ["--stride", str(stride)]
    if cutoff is not None:
        cmd += ["--consensus-cutoff", str(cutoff)]
    if merge:
        cmd += ["--merge", merge, "--merge-iso", str(merge_iso)]
    if holos:
        cmd += ["--holo", *map(str, holos)]
    start = time.time()
    with open(where / "cgpocket.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return model, where.name, where, code, time.time() - start


def site_numbers(table: Path):
    f = np.array([float(r["pocket_frequency_within_2A"]) for r in csv.DictReader(open(table))])
    return float(np.median(f)), float(np.mean(f >= 0.5))


def frames_used(log: Path) -> str:
    for line in open(log):
        if line.startswith("wrote ") and " frames" in line:
            return line.rsplit(",", 1)[-1].split()[0]
    return ""


RANKINGS = ("quality", "persistence", "quality_burial")
#: PPc_core: the same test from the centre of the pocket's enclosed core
CRITERIA = ("PPc", "MOc", "PPc_core")


def first_ranks(table: Path) -> dict:
    """First PPc / MOc rank of the holo site under each ranking, and whether that
    pocket is flagged cryptic."""
    rows = list(csv.DictReader(open(table)))
    out = {}
    for r in RANKINGS:
        for crit in CRITERIA:
            hits = [int(x[f"rank_{r}"]) for x in rows if x.get(crit) == "True"]
            out[f"{r}_{crit}"] = min(hits) if hits else None
    site = [x for x in rows if x["PPc"] == "True"]
    best = min(site, key=lambda x: int(x["rank_quality"])) if site else None
    out["site_cryptic"] = best["cryptic"] if best else ""
    out["site_occupancy"] = best["occupancy"] if best else ""
    return out


def summarize(model_dir: Path, rows) -> None:
    rank_cols = [f"{r}_{c}" for r in RANKINGS for c in CRITERIA]
    lines = [["apo", "holo", "frames", "site_frequency_median", "ligand_atoms_in_pocket_half_the_frames",
              *[f"first_rank_{c}" for c in rank_cols], "site_occupancy", "site_cryptic",
              "seconds", "status"]]  # fmt: skip
    tops = {c: [] for c in rank_cols}
    for pair, where, code, seconds, holos in sorted(rows):
        apo = pair.split("_")[0]
        frames = frames_used(where / "cgpocket.log") if (where / "cgpocket.log").exists() else ""
        status = "ok" if code == 0 else "failed (see cgpocket.log)"
        for h in holos:
            tag = h.name.split(".")[0]
            table = where / f"ligand_site_frequency_{tag}.csv"
            med, frac = site_numbers(table) if table.exists() else (float("nan"), float("nan"))
            ranked = where / "pockets_vs_holo.csv"
            fr = first_ranks(ranked) if ranked.exists() else {}
            for c in rank_cols:
                tops[c].append(fr.get(c))
            lines.append([apo, tag, frames, f"{med:.2f}", f"{frac:.2f}",
                          *[fr.get(c) or "" for c in rank_cols], fr.get("site_occupancy", ""),
                          fr.get("site_cryptic", ""), f"{seconds:.0f}", status])  # fmt: skip
    with open(model_dir / "summary.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(lines)
    body = [r for r in lines[1:] if r[1] and r[3] not in ("", "nan")]
    med = [float(r[3]) for r in body]
    md = [f"# {model_dir.name}: mdpocket on {len(rows)} trajectories", "",
          "For each apo/holo pair: how often the holo ligand's site is a pocket over the trajectory",
          "(median over ligand atoms of the largest pocket frequency within 2 Å).", "",
          f"- pairs: {len(body)}; site open in at least half the frames (median >= 0.5): "
          f"{sum(m >= 0.5 for m in med)}; median of the medians: {np.median(med):.2f}" if med else "- no pairs",
          "", "Consensus-pocket ranking over the trajectory (cgpocket.py traj --rank): fraction of",
          "pairs with a correct pocket among the first k, Top-1 / Top-3 / Top-5 / Top-10.", "",
          "| ranking | PPc | MOc | PPc, core centre |", "|---|---|---|---|",
          *[f"| {r.replace('_', ' x ')} | " + " | ".join(
              " / ".join(f"{sum(1 for x in tops[f'{r}_{c}'] if x and x <= k) / max(len(tops[f'{r}_{c}']), 1):.2f}"
                         for k in (1, 3, 5, 10)) for c in CRITERIA) + " |" for r in RANKINGS],
          "", "Per pair: summary.csv; per ligand atom: <apo>_<holo>/ligand_site_frequency_<holo>.csv;",
          "view: cd <apo>_<holo> && pymol view.pml"]  # fmt: skip
    (model_dir / "summary.md").write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the run directory: <model>/<apo>/...")
    ap.add_argument("--data", type=Path, required=True, help="where apo/, holo/ and si.csv are")
    ap.add_argument("--out", type=Path, help="default: <root>/fpocket")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--consensus-cutoff", type=float, help="passed to cgpocket.py traj (default 6 Å)")
    ap.add_argument("--merge", choices=("centroid", "map"), help="passed to cgpocket.py traj")
    ap.add_argument("--merge-iso", type=float, default=0.2, help="passed to cgpocket.py traj")
    args = ap.parse_args()
    out = args.out or args.root / "fpocket"
    pairs = defaultdict(list)  # apo -> [(holo file, apo file)]
    for p in prepare.schrodinger_pairs(args.data):
        pairs[p["apo"]].append((Path(p["holo_file"]), Path(p["apo_file"])))
    runs = find_runs(args.root, args.models)
    jobs = []
    for model, apo, md in runs:
        if not pairs.get(apo):
            print(f"skip {model}/{apo}: no apo/holo pair in si.csv")
            continue
        for holo_file, apo_file in pairs[apo]:
            jobs.append((model, apo, md, apo_file, [holo_file], out, args.stride, args.consensus_cutoff,
                         args.merge, args.merge_iso))
    with_traj = {(m, a) for m, a, _ in runs}
    for apo, plist in pairs.items():
        for model in args.models:
            if (model, apo) not in with_traj:
                for holo_file, _ in plist:
                    print(f"skip {model}/{apo}_{holo_file.stem}: no trajectory")
    print(f"{len(jobs)} apo/holo entries, stride {args.stride}", flush=True)
    done = defaultdict(list)
    with ProcessPoolExecutor(args.jobs) as pool:
        for job, (model, pair, where, code, seconds) in zip(jobs, pool.map(run_one, jobs), strict=True):
            done[model].append((pair, where, code, seconds, job[4]))
            if code:
                print(f"FAILED {model}/{pair}: {where / 'cgpocket.log'}", flush=True)
    for model in args.models:
        if done[model]:
            summarize(out / model, done[model])
            print(f"{model}: {sum(1 for r in done[model] if r[2] == 0)}/{len(done[model])} entries ok; "
                  + (out / model / "summary.md").read_text().splitlines()[5])


if __name__ == "__main__":
    main()
