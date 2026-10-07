"""Volume overlap of consensus pockets with the holo ligand, each pocket on its own best frame.

For every <model>/<apo>_<holo>/ of a traj --rank output (it needs
frame_pockets.npz), the top-K consensus pockets of a ranking are taken one at a
time. Each pocket is its best frame's alpha spheres; the holo structure is
superposed on that very frame (cgpocket.place_holo, whole-protein fit), so the
ligand follows the protein rather than sitting where it was in the first frame.
Then, per pocket:

  PPc_frame        best frame's sphere centroid < 4 Å from that frame's ligand
  volume measures  cg/volume.py: the alpha spheres' empty space (r - 1.7 Å) and
                   the ligand on a common 1 Å grid -> ligand_volume_covered,
                   pocket_volume_in_ligand, pocket_volume_near_ligand (2 Å), DVO

Writes <pair>/volume_vs_holo.csv and prints, per model, Top-1/3/5/10 by
PPc_frame and the volume measures of the first PPc_frame pocket.

    python volume_overlap.py ~/Dropbox/PocketFinding/SchrodingerSet/20261005_apo/fpocket -j 12
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
import volume as V  # noqa: E402

HOLO = Path.home() / "Dropbox/PocketFinding/SchrodingerSet/holo"
MEASURES = ["ligand_volume_covered", "pocket_volume_in_ligand", "pocket_volume_near_ligand", "DVO"]


def frames(md: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    import boonza

    beads = C.load(md.with_suffix(".pdb"))
    out = {}
    for f, frame in enumerate(boonza.open_trajectory(str(md), beads)):
        if f in wanted:
            out[f] = np.asarray(frame.positions, float)
            if len(out) == len(wanted):
                break
    return out


def one(args):
    where, ranking, k, shrink = args
    import boonza

    holo_id = where.name.split("_", 1)[1]
    holo = boonza.load(str(HOLO / f"{holo_id}.mae"))
    rows = sorted(csv.DictReader(open(where / "pockets_vs_holo.csv")), key=lambda r: int(r[f"rank_{ranking}"]))[:k]
    z = np.load(where / "frame_pockets.npz")
    md = where / "mdpocket" / "md.dcd"
    xyz = frames(md, {int(r["best_frame"]) for r in rows})
    beads = C.load(md.with_suffix(".pdb"))
    ligands = {}
    out = []
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
        dca = float(np.linalg.norm(lig - centres.mean(0), axis=1).min())
        out.append({"rank": int(r[f"rank_{ranking}"]), "rank_quality": r["rank_quality"], "best_frame": f,
                    "PPc_first_frame": r["PPc"], "center_to_ligand_frame": round(dca, 3),
                    "PPc_frame": dca < C.PPC_CUTOFF,
                    **{k2: round(v, 4) for k2, v in V.overlap(V.sphere_pocket(centres, radii, shrink), lig).items()}})  # fmt: skip
    with open(where / "volume_vs_holo.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return where.parent.name, where.name, out


def tops(ranks, n):
    return " / ".join(f"{sum(1 for x in ranks if x and x <= k) / n:.2f}" for k in (1, 3, 5, 10))


def spread(v):
    return f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f} (median {statistics.median(v):.2f})" if len(v) > 1 else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a traj --rank output: <root>/<model>/<apo>_<holo>/")
    ap.add_argument("--models", nargs="+", default=["martini2", "martini3", "sirah"])
    ap.add_argument("--ranking", default="quality_burial", choices=["quality", "persistence", "quality_burial"])
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--shrink", type=float, default=V.SPHERE_SHRINK, help="Å taken off each alpha sphere")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    args = ap.parse_args()
    jobs = [(d, args.ranking, args.k, args.shrink) for m in args.models for d in sorted((args.root / m).iterdir())
            if (d / "frame_pockets.npz").exists()]  # fmt: skip
    by_model = {}
    with ProcessPoolExecutor(args.jobs) as pool:
        for model, pair, out in pool.map(one, jobs):
            by_model.setdefault(model, {})[pair] = out
    for model, pairs in by_model.items():
        n = len(pairs)
        first = {p: next((r for r in o if r["PPc_frame"]), None) for p, o in pairs.items()}
        old = {p: next((r["rank"] for r in o if r["PPc_first_frame"] == "True"), None) for p, o in pairs.items()}
        print(f"\n== {model}: {n} pairs, top {args.k} by {args.ranking}")
        print(f"  PPc, ligand placed on frame 0 : {tops(old.values(), n)}")
        print(f"  PPc, ligand on the best frame : {tops([r['rank'] if r else None for r in first.values()], n)}")
        for m in MEASURES:
            print(f"  first PPc pocket {m:26s} {spread([r[m] for r in first.values() if r])}")
        print(f"  best DVO among the top {args.k:<2d}           {spread([max(r['DVO'] for r in o) for o in pairs.values()])}")


if __name__ == "__main__":
    main()
