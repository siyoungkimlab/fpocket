"""Score fpocket on a benchmark set, for one representation and one set of flags.

The truth for each structure is the holo ligand's heavy atoms, in the
structure's frame (see prepare.py), and its binding site as residues.  The
two criteria of the fpocket paper, as tpocket implements them:

* PPc: a pocket is *correct* when its center -- the mean of its alpha-sphere
  centers -- is less than ``DCA_CUTOFF`` from a ligand heavy atom;
* MOc (mutual overlap): more than ``MOC_LIGAND`` of the ligand's atoms lie
  within ``MOC_D`` of the pocket's sphere centers, and more than
  ``MOC_POCKET`` of its sphere centers lie within ``MOC_D`` of the ligand.

The rank of the first correct pocket is kept for every structure; Top-k is
the fraction of structures with a correct pocket among the first k (PPc;
``*_moc`` for MOc).  A
residue-overlap criterion is kept alongside (``*_res``): at least
``PRECISION`` of the residues lining the pocket (those of the atoms or beads
its spheres touch) are site residues, and the pocket lines at least
``COVERAGE`` of the site.

    python evaluate.py --set pp48 --model martini3 -- -m 3.4 -M 6.2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import importlib

import numpy as np
from scipy.spatial import cKDTree

boonza_sites = importlib.import_module("boonza.sites")  # boonza.sites is also a function

HERE = Path(__file__).resolve().parent
FPOCKET = HERE.parents[1] / "bin" / "fpocket"
DATA = HERE / "data"


def presets_path() -> Path:
    """The presets file a search writes: boonza's own, data/pockets/presets.json in the boonza
    checkout installed here (commit it there, through a pull request), or $BOONZA_PRESETS."""
    import os

    if os.environ.get("BOONZA_PRESETS"):
        return Path(os.environ["BOONZA_PRESETS"])
    from boonza.pockets.run import PRESETS

    return Path(PRESETS)
SETS = ("pp48", "schrodinger", "train263")

PRECISION = 0.5
COVERAGE = 0.2
DCA_CUTOFF = 4.0  # Å: PPc, pocket center to ligand (tpocket M_CRIT3_VAL)
MOC_D = 3.0  # Å: MOc contact distance (tpocket M_CRIT4_D, M_CRIT5_D)
MOC_LIGAND = 0.5  # MOc: share of ligand atoms near the pocket (M_CRIT4_VAL)
MOC_POCKET = 0.2  # MOc: share of pocket spheres near the ligand (M_CRIT5_VAL)
#: the descriptors fpocket writes with --write_score_descriptors, in order
SCORE_FEATURES = ("nas_norm", "prop_asapol_norm", "mean_loc_hyd_dens_norm", "polarity_score",
                  "as_density", "convex_hull_volume", "surf_pol_vdw14", "surf_apol_vdw14")
TOUCH = 0.25  # Å of slack when deciding which atoms a sphere touches
TIMEOUT = 20  # s per structure


@lru_cache(maxsize=None)
def read_pdb(path: str):
    xyz, labels = [], []
    for line in open(path):
        if line.startswith("ATOM"):
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            labels.append((line[21], int(line[22:26]), line[26].strip()))
    return np.array(xyz), labels


def structures(dataset: str, kinds=("apo", "holo")) -> list[dict]:
    """One entry per (structure, site): what each fpocket run is scored against."""
    entries = json.loads((DATA / dataset / "cases.json").read_text())["structures"]
    out = []
    for e in entries:
        if e.get("exclude"):
            continue
        if e["kind"] in kinds and (DATA / dataset / "pdb" / f"{e['stem']}_aa.pdb").exists():
            out.append({**e, "set": dataset, "site": [tuple(x) for x in e["site"]]})
    return out


def site_residues(labels, site):
    """The residues of ``labels`` that are in ``site``; chains ignored when the
    site names chains the structure does not have (they were relabeled)."""
    chains = {c for c, _, _ in labels}
    use_chain = {c for c, _, _ in site} <= chains
    key = (lambda r: r) if use_chain else (lambda r: r[1:])
    wanted = {key(r) for r in site}
    return {r for r in set(labels) if key(r) in wanted}


def run_fpocket(pdb: Path, flags: list[str]):
    """The pockets fpocket finds, in its ranking: [(centers, radii, score descriptors)].

    The descriptors are those of --write_score_descriptors when ``flags`` ask for
    them, else None."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("FPOCKET_TMP")) as tmp:
        local = Path(tmp) / pdb.name
        os.symlink(pdb.resolve(), local)
        try:
            subprocess.run([str(FPOCKET), "-f", str(local), *flags], check=True, timeout=TIMEOUT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        except subprocess.TimeoutExpired:
            return []  # too slow to be usable: as if nothing was found
        pqr = Path(tmp) / f"{pdb.stem}_out" / f"{pdb.stem}_pockets.pqr"
        desc = Path(tmp) / f"{pdb.stem}_out" / f"{pdb.stem}_score_desc.txt"
        features = {}
        if desc.exists():
            for line in list(open(desc))[1:]:
                w = line.split()
                features[int(w[0])] = [float(x) for x in w[2:]]
        pockets: dict[int, list] = {}
        if pqr.exists():
            for line in open(pqr):
                if line.startswith("ATOM"):
                    k = int(line[22:26])
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    r = float(line[66:].split()[0]) if len(line) > 66 else 0.0
                    pockets.setdefault(k, []).append((x, y, z, r))
    return [(np.array(v)[:, :3], np.array(v)[:, 3], features.get(k))
            for k, v in sorted(pockets.items())]  # fmt: skip


def mutual_overlap(centers, ligand) -> tuple[float, float]:
    """(share of ligand atoms within MOC_D of a sphere center, share of sphere
    centers within MOC_D of a ligand atom)."""
    d = np.linalg.norm(ligand[:, None] - centers[None], axis=2)
    near = d < MOC_D
    return float(near.any(1).mean()), float(near.any(0).mean())


def score_structure(args):
    entry, model, flags, max_rank = args
    pdb = DATA / entry["set"] / "pdb" / f"{entry['stem']}_{model}.pdb"
    xyz, labels = read_pdb(str(pdb))
    aa_xyz, aa_labels = read_pdb(str(DATA / entry["set"] / "pdb" / f"{entry['stem']}_aa.pdb"))
    ligand = np.array(entry["ligand"])
    site_labels = entry["site"]
    site = site_residues(labels, site_labels)
    aa_site = site_residues(aa_labels, site_labels)
    site_center = aa_xyz[[lab in aa_site for lab in aa_labels]].mean(0)
    tree = cKDTree(xyz)
    pockets = run_fpocket(pdb, flags)
    rows, table = [], []
    for centers, radii, feats in pockets[:max_rank]:
        touched = set()
        for c, r in zip(centers, radii, strict=True):
            touched.update(tree.query_ball_point(c, r + TOUCH))
        lining = {labels[i] for i in touched}
        common = len(lining & site)
        center = centers.mean(0)
        dca = boonza_sites.dca(centers, ligand)  # pocket center to nearest ligand atom
        lig_cov, pock_cov = mutual_overlap(centers, ligand)
        ppc, moc = dca < DCA_CUTOFF, lig_cov > MOC_LIGAND and pock_cov > MOC_POCKET
        if feats is not None:
            table.append(feats + [float(ppc), float(moc)])
        rows.append({
            "precision": common / max(len(lining), 1),
            "coverage": common / max(len(site), 1),
            "dcc": float(np.linalg.norm(center - site_center)),
            "dca": dca,
            "ppc": bool(ppc),
            "moc": bool(moc),
            "nsph": len(centers),
            "nres": len(lining),
        })  # fmt: skip
    hit = next((k + 1 for k, r in enumerate(rows) if is_hit(r)), None)
    hit_dca = next((k + 1 for k, r in enumerate(rows) if r["ppc"]), None)
    hit_moc = next((k + 1 for k, r in enumerate(rows) if r["moc"]), None)
    out = {"case": entry["case"], "kind": entry["kind"], "stem": entry["stem"],
           "npockets": len(pockets), "hit_rank": hit, "hit_rank_dca": hit_dca,
           "hit_rank_moc": hit_moc, "pockets": rows[:5]}  # fmt: skip
    if table:  # every pocket's score descriptors, then its PPc and MOc labels
        out["table"] = table
    return out


def is_hit(row, precision=PRECISION, coverage=COVERAGE) -> bool:
    return row["precision"] >= precision and row["coverage"] >= coverage


def summarize(results) -> dict:
    n = len(results)

    def top(k, key="hit_rank_dca"):
        return sum(1 for r in results if r[key] is not None and r[key] <= k) / n

    first = [r["pockets"][0] for r in results if r["pockets"]]
    return {"n": n, "top1": top(1), "top3": top(3), "top5": top(5), "top10": top(10),
            "found": top(10**6),
            "top1_moc": top(1, "hit_rank_moc"), "top3_moc": top(3, "hit_rank_moc"),
            "top5_moc": top(5, "hit_rank_moc"), "top10_moc": top(10, "hit_rank_moc"),
            "top1_res": top(1, "hit_rank"), "top3_res": top(3, "hit_rank"),
            "top5_res": top(5, "hit_rank"),
            "mean_pockets": float(np.mean([r["npockets"] for r in results])),
            "top1_mean_res": float(np.mean([p["nres"] for p in first])) if first else 0.0}  # fmt: skip


def evaluate(dataset: str, model: str, flags: list[str], kinds=("apo", "holo"), jobs: int = 10,
             max_rank: int = 1000, pool=None) -> tuple[dict, list]:  # fmt: skip
    entries = structures(dataset, kinds)
    work = [(e, model, list(flags), max_rank) for e in entries]
    if pool is not None:
        results = list(pool.map(score_structure, work))
    else:
        with ProcessPoolExecutor(jobs) as p:
            results = list(p.map(score_structure, work))
    out = {}
    for kind in kinds:
        out[kind] = summarize([r for r in results if r["kind"] == kind])
    out["all"] = summarize(results)
    return out, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--set", default="pp48", choices=SETS)
    ap.add_argument("--model", default="aa", help="aa, martini2, martini3, sirah, martini3_npolar ...")
    ap.add_argument("--kinds", default="apo,holo")
    ap.add_argument("-j", "--jobs", type=int, default=10)
    ap.add_argument("--detail", action="store_true", help="print every structure")
    ap.add_argument("--json", type=Path, help="write per-structure results here")
    ap.add_argument("flags", nargs=argparse.REMAINDER, help="fpocket flags, after --")
    args = ap.parse_args()
    flags = [f for f in args.flags if f != "--"]
    if not shutil.which(str(FPOCKET)):
        sys.exit(f"no fpocket at {FPOCKET}")
    summary, results = evaluate(args.set, args.model, flags, tuple(args.kinds.split(",")), args.jobs)
    if args.detail:
        for r in results:
            first = r["pockets"][0] if r["pockets"] else None
            extra = f"top pocket: center {first['dca']:.1f} Å from the ligand" if first else ""
            print(f"{r['kind']:4s} {r['case']:10s} pockets {r['npockets']:3d} "
                  f"correct at rank {r['hit_rank_dca']}  {extra}")  # fmt: skip
    for k, v in summary.items():
        print(f"{k:4s} " + "  ".join(f"{a} {b:.3f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()))
    if args.json:
        args.json.write_text(json.dumps({"model": args.model, "flags": flags, "summary": summary,
                                         "results": results}, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
