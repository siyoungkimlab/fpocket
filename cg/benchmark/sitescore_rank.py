"""Rank consensus pockets by a SiteScore-like score of their enclosed cores (an experiment).

SiteScore (Halgren 2009, eq 3) = 0.0733 sqrt(n) + 0.6688 e - 0.20 p, with n the
site points (capped at 100), e the enclosure (share of 110 rays that meet the
receptor within 10 Å, averaged over the site points) and p the hydrophilic
score; Halgren notes the p term is not needed to find sites, so it is left out.
For each consensus pocket of a traj --rank output, on its best frame:

  n  core volume / 2.2 Å^3 (SiteMap places ~0.45 site points per Å^3 of its volume)
  e  share of 60 rays from each core point that meet a bead (within its radius)
     inside 10 Å, averaged over the core

The core is core.py's, as in consensus_cores.pqr. Pockets open in fewer than 5%
of frames rank last, as for quality. Success: PPc from the core centre against the
ligand of pockets_vs_holo.csv. Nothing is fitted.

    python sitescore_rank.py ~/Dropbox/PocketFinding/SchrodingerSet/20261005_apo/fpocket -j 12
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cgpocket as C  # noqa: E402
import core as K  # noqa: E402
from volume_overlap import frames  # noqa: E402

#: SiteScore-like weights refitted on fpocket's training set (sitescore_fit.py)
FITTED_PATH = Path(__file__).resolve().parent / "runs" / "sitescore" / "fitted.json"
FITTED = __import__("json").loads(FITTED_PATH.read_text()) if FITTED_PATH.exists() else {}
POINTS_PER_A3 = 0.45
N_CAP = 100
RAY = 10.0


def enclosure(points, beads, radii, n_rays=60):
    tree = cKDTree(beads)
    dirs = K._directions(n_rays)
    steps = np.arange(1.0, RAY + 0.5, 1.0)
    probes = (points[:, None, None, :] + dirs[None, :, None, :] * steps[None, None, :, None]).reshape(-1, 3)
    d, j = tree.query(probes, k=4)
    return float((d <= radii[j]).any(1).reshape(len(points), n_rays, len(steps)).any(2).mean())


def one(where: Path):
    import boonza

    model = where.parent.name
    rows = list(csv.DictReader(open(where / "pockets_vs_holo.csv")))
    lig_sys = boonza.load(str(next(where.glob("ligand_*.mae"))))
    lig = np.asarray(lig_sys.positions)[[a > 1 for a in lig_sys.atoms["anum"]]]
    run = where.parents[2] / model / where.name.split("_", 1)[0]
    dms = boonza.load(str(next(run.glob("*/sim_*/md_solute/solvated.dms"))))
    radii = K.bead_radii(np.asarray(dms.atoms["type"])[C.protein_ids(dms)], model)
    z = np.load(where / "frame_pockets.npz")
    xyz = frames(where / "mdpocket" / "md.dcd", {int(r["best_frame"]) for r in rows})
    out = []
    for r in rows:
        f, q = int(r["best_frame"]), int(r["rank_quality"])
        members = np.flatnonzero((z["consensus"] == q) & (z["frame"] == f))
        i = members[np.argmax(z["p"][members])]
        a, b = z["offsets"][i], z["offsets"][i + 1]
        pts = K.core_points(z["sphere_centers"][a:b], z["sphere_radii"][a:b], xyz[f], radii)
        v = len(pts) * K.SPACING**3
        s_rep = float(z["score"][i])
        if len(pts):
            n = min(v * POINTS_PER_A3, N_CAP)
            e = enclosure(pts, xyz[f], radii)
            score = 0.0733 * np.sqrt(n) + 0.6688 * e
            ppc = float(np.linalg.norm(lig - pts.mean(0), axis=1).min()) < C.PPC_CUTOFF
        else:
            n = e = 0.0
            score, ppc = -np.inf, False
        g = FITTED.get(model)
        fit_a = np.sqrt(min(v, g["A"][0])) + g["A"][1] * e if g else np.nan
        fit_b = s_rep + g["B"][1] * np.sqrt(min(v, g["B"][0])) + g["B"][2] * e if g else np.nan
        out.append({"rank_quality": r["rank_quality"],
                    "rank_quality_burial": int(r["rank_quality_burial"]), "occupancy": float(r["occupancy"]),
                    "sitescore": score, "fit_a": fit_a, "fit_b": fit_b, "s_best_frame": s_rep,
                    "n": n, "e": e, "PPc_core": ppc})  # fmt: skip
    return model, where.name, out


def first(rows, key, floor):
    order = sorted(rows, key=lambda r: ((floor and r["occupancy"] < C.MIN_OCCUPANCY), -r[key]) if key != "rank_quality_burial"
                   else r[key])  # fmt: skip
    return next((k for k, r in enumerate(order, 1) if r["PPc_core"]), None)


def tops(v):
    return " / ".join(f"{sum(1 for x in v if x and x <= k) / len(v):.2f}" for k in (1, 3, 5, 10))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--models", nargs="+", default=["martini2", "martini3", "sirah"])
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--dump", help="write every pocket's scores to this JSON")
    args = ap.parse_args()
    dirs = [d for m in args.models for d in sorted((args.root / m).iterdir()) if (d / "frame_pockets.npz").exists()]
    res = {}
    with ProcessPoolExecutor(args.jobs) as pool:
        for model, pair, out in pool.map(one, dirs):
            res.setdefault(model, {})[pair] = out
    if args.dump:
        Path(args.dump).write_text(__import__("json").dumps({m: p for m, p in res.items()}, default=float))
    for model, pairs in res.items():
        print(f"\n== {model} ({len(pairs)} pairs), PPc from the core centre, Top-1/3/5/10")
        print(f"  quality x burial (current)       {tops([first(o, 'rank_quality_burial', False) for o in pairs.values()])}")
        print(f"  SiteScore-like, 5% floor         {tops([first(o, 'sitescore', True) for o in pairs.values()])}")
        print(f"  SiteScore-like, no floor         {tops([first(o, 'sitescore', False) for o in pairs.values()])}")
        print(f"  fpocket score of best frame      {tops([first(o, 's_best_frame', True) for o in pairs.values()])}")
        print(f"  refit A (SiteScore form)         {tops([first(o, 'fit_a', True) for o in pairs.values()])}")
        print(f"  refit B (score + SiteMap terms)  {tops([first(o, 'fit_b', True) for o in pairs.values()])}")
        top = [max(o, key=lambda r: (r["occupancy"] >= C.MIN_OCCUPANCY, r["sitescore"])) for o in pairs.values()]
        print(f"  top site: n median {np.median([t['n'] for t in top]):.0f}, enclosure median {np.median([t['e'] for t in top]):.2f}")


if __name__ == "__main__":
    main()
