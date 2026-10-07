"""Volume overlap of the enclosed cores of consensus pockets with the holo ligand.

As volume_overlap.py (top-K consensus pockets of a ranking, each on its best
frame, the holo ligand superposed on that frame), but each pocket is first
trimmed to its enclosed core (cg/core.py, after SiteMap's site-point rules).
The pocket ranking is unchanged. Writes <pair>/core_vs_holo.csv:

  core_points, core_volume       the core (0 if nothing in the pocket is enclosed)
  PPc_core / PPc_pocket          centroid of the core / of the full pocket (best frame)
                                 < 4 Å from that frame's ligand
  ligand_volume_covered, pocket_volume_near_ligand, pocket_volume_in_ligand, DVO
                                 of the core (cg/volume.py)

    python core_overlap.py ~/Dropbox/PocketFinding/SchrodingerSet/20261005_apo/fpocket -j 12
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cgpocket as C  # noqa: E402
import core as K  # noqa: E402
import volume as V  # noqa: E402
from volume_overlap import HOLO, frames  # noqa: E402

MEASURES = ["ligand_volume_covered", "pocket_volume_near_ligand", "pocket_volume_in_ligand", "DVO", "core_volume"]


def one(args):
    where, ranking, k, spacing, outside = args
    K.OUTSIDE = outside  # worker processes start with the module default
    import boonza

    model = where.parent.name
    run = where.parents[2] / model / where.name.split("_", 1)[0]
    dms = boonza.load(str(next(run.glob("*/sim_*/md_solute/solvated.dms"))))
    bead_radii = K.bead_radii(np.asarray(dms.atoms["type"])[C.protein_ids(dms)], model)
    holo = boonza.load(str(HOLO / f"{where.name.split('_', 1)[1]}.mae"))
    rows = sorted(csv.DictReader(open(where / "pockets_vs_holo.csv")), key=lambda r: int(r[f"rank_{ranking}"]))[:k]
    z = np.load(where / "frame_pockets.npz")
    md = where / "mdpocket" / "md.dcd"
    xyz = frames(md, {int(r["best_frame"]) for r in rows})
    beads = C.load(md.with_suffix(".pdb"))
    if len(bead_radii) != beads.natoms:
        sys.exit(f"{where}: {len(bead_radii)} bead radii for {beads.natoms} beads")
    ligands, out = {}, []
    for r in rows:
        f = int(r["best_frame"])
        if f not in ligands:
            ref = beads.clone()
            ref.positions = xyz[f]
            ligands[f] = C.place_holo(holo, ref, "resname LIG")[0]
        lig = ligands[f]
        members = np.flatnonzero((z["consensus"] == int(r["rank_quality"])) & (z["frame"] == f))
        i = members[np.argmax(z["p"][members])]
        a, b = z["offsets"][i], z["offsets"][i + 1]
        centres, radii = z["sphere_centers"][a:b], z["sphere_radii"][a:b]
        pts = K.core_points(centres, radii, xyz[f], bead_radii, spacing)
        dca = lambda c: float(np.linalg.norm(lig - c, axis=1).min())  # noqa: E731
        row = {"rank": int(r[f"rank_{ranking}"]), "rank_quality": r["rank_quality"], "best_frame": f,
               "PPc_pocket": dca(centres.mean(0)) < C.PPC_CUTOFF, "core_points": len(pts),
               "core_volume": len(pts) * spacing**3,
               "PPc_core": bool(len(pts)) and dca(pts.mean(0)) < C.PPC_CUTOFF}  # fmt: skip
        if len(pts):
            ov = V.overlap(V.grid_pocket(pts, spacing), lig)
            row.update({m: round(ov[m], 4) for m in MEASURES[:4]})
        else:
            row.update({m: 0.0 for m in MEASURES[:4]})
        out.append(row)
    with open(where / "core_vs_holo.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return model, where.name, out


def tops(ranks, n):
    return " / ".join(f"{sum(1 for x in ranks if x and x <= k) / n:.2f}" for k in (1, 3, 5, 10))


def spread(v):
    return f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f} (median {statistics.median(v):.2f})" if len(v) > 1 else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--models", nargs="+", default=["martini2", "martini3", "sirah"])
    ap.add_argument("--ranking", default="quality_burial", choices=["quality", "persistence", "quality_burial"])
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--spacing", type=float, default=K.SPACING)
    ap.add_argument("--outside", type=float, default=K.OUTSIDE,
                    help="a core point must be this many bead radii from every bead (SiteMap: 1.58)")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    args = ap.parse_args()
    K.OUTSIDE = args.outside
    jobs = [(d, args.ranking, args.k, args.spacing, args.outside) for m in args.models for d in sorted((args.root / m).iterdir())
            if (d / "frame_pockets.npz").exists()]  # fmt: skip
    by_model = {}
    with ProcessPoolExecutor(args.jobs) as pool:
        for model, pair, out in pool.map(one, jobs):
            by_model.setdefault(model, {})[pair] = out
    for model, pairs in by_model.items():
        n = len(pairs)
        print(f"\n== {model}: {n} pairs, top {args.k} by {args.ranking}, grid {args.spacing} Å")
        for key in ("PPc_pocket", "PPc_core"):
            print(f"  {key:10s} Top-1/3/5/10 {tops([next((r['rank'] for r in o if r[key]), None) for o in pairs.values()], n)}")
        empty = sum(1 for o in pairs.values() for r in o if not r["core_points"]) / sum(len(o) for o in pairs.values())
        print(f"  share of top-{args.k} pockets with no enclosed core: {empty:.2f}")
        first = [next((r for r in o if r["PPc_core"]), None) for o in pairs.values()]
        for m in MEASURES:
            print(f"  first PPc_core site {m:26s} {spread([r[m] for r in first if r])}")


if __name__ == "__main__":
    main()
