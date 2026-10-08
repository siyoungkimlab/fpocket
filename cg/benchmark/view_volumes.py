"""PyMOL views of pocket volumes: a consensus pocket (best frame) and an MxMD + SiteMap site, with the ligand.

For each <apo>_<holo> pair: our first PPc pocket (volume_vs_holo.csv, ligand on
its best frame) and SiteMap's first PPc site (MxMD_SiteMap CSV). SiteMap's site
comes from a different MD frame, so its site points and frame are moved onto
our frame by fitting its copy of the holo ligand onto ours (the same ligand,
superposed on each frame by results.ligand_on, whole-protein fit).

Writes <out>/<pair>/: view.pml, apo.mae (all-atom apo, our frame), ligand.pdb,
fpocket_volume.pqr (alpha spheres at r - 1.7 Å: the volume measured),
fpocket_spheres.pqr (full alpha spheres), sitemap_volume.pqr (site points at
0.85 Å: the volume measured), sitemap_frame.pdb (the SiteMap MD frame, moved).

    python view_volumes.py 1urp_2dri 4i92_4i94 --model martini3 \\
        --traj ~/Dropbox/PocketFinding/SchrodingerSet/20261005_apo/fpocket \\
        --out ~/Dropbox/PocketFinding/SchrodingerSet/volume_view
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import boonza
from boonza.pockets import overlap as V

import results as R

DATA = Path.home() / "Dropbox/PocketFinding/SchrodingerSet"
sys.path.insert(0, str(DATA / "MxMD_SiteMap"))


def kabsch(a: np.ndarray, b: np.ndarray):
    """Rotation r and translation t with a @ r + t ~= b."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return r, cb - ca @ r


def write_pqr(path: Path, xyz, radii, resname: str) -> None:
    with open(path, "w") as fh:
        for i, ((x, y, z), r) in enumerate(zip(xyz, radii, strict=True), 1):
            fh.write(f"ATOM  {i:5d}  C   {resname} A   1    {x:8.3f}{y:8.3f}{z:8.3f} {0:7.4f} {r:6.3f}\n")


def write_ligand(path: Path, xyz) -> None:
    with open(path, "w") as fh:
        for i, (x, y, z) in enumerate(xyz, 1):
            fh.write(f"HETATM{i:5d}  C{i % 100:<2d} LIG L   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")


def ours(where: Path, model_dir: Path):
    """Best-frame alpha spheres and frame-placed ligand of our first PPc pocket."""

    rows = list(csv.DictReader(open(where / "volume_vs_holo.csv")))
    r = next(x for x in rows if x["PPc_frame"] == "True")
    f, q = int(r["best_frame"]), int(r["rank_quality"])
    z = np.load(where / "frame_pockets.npz")
    members = np.flatnonzero((z["consensus"] == q) & (z["frame"] == f))
    i = members[np.argmax(z["p"][members])]
    a, b = z["offsets"][i], z["offsets"][i + 1]
    md = R.md(where)
    beads = boonza.load(md.with_suffix(".pdb"))
    for k, frame in enumerate(boonza.open_trajectory(str(md), beads)):
        if k == f:
            beads.positions = np.asarray(frame.positions, float)
            break
    holo = boonza.load(str(DATA / "holo" / f"{where.name.split('_', 1)[1]}.mae"))
    lig = R.ligand_on(holo, beads)
    return r, z["sphere_centers"][a:b], z["sphere_radii"][a:b], lig


def sitemap(pair: str):
    """Site points, frame and frame-placed ligand of SiteMap's first PPc site."""
    import sitemap_vs_holo as M

    row = next(x for x in csv.DictReader(open(DATA / "MxMD_SiteMap/mxmd_sitemap_result" / f"{pair}.csv"))
               if x["PPc"] == "True")  # fmt: skip
    rank, seen, frame = int(row["rank"]), 0, None
    with tempfile.TemporaryDirectory() as tmp:
        for k, (header, block) in enumerate(M.cts(DATA / "MxMD_SiteMap/mxmd_sitemap" / f"{pair}.maegz")):
            if "r_sitemap_SiteScore" not in M.properties(block):
                frame = M.load_block(header, block, Path(tmp), k)
                continue
            seen += 1
            if seen == rank:
                points = np.asarray(M.load_block(header, block, Path(tmp), k).positions)
                break
    holo = boonza.load(str(DATA / "holo" / f"{pair.split('_', 1)[1]}.mae"))
    lig = R.ligand_on(holo, frame)
    return row, points, frame, lig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--model", default="martini3")
    ap.add_argument("--traj", type=Path, default=DATA / "20261005_apo/fpocket")
    ap.add_argument("--out", type=Path, default=DATA / "volume_view")
    args = ap.parse_args()
    from boonza.io import save

    for pair in args.pairs:
        where = args.traj / args.model / pair
        out = args.out / f"{pair}_{args.model}"
        out.mkdir(parents=True, exist_ok=True)
        r, centres, radii, lig = ours(where, args.traj / args.model)
        srow, points, sframe, slig = sitemap(pair)
        rot, t = kabsch(slig, lig)  # SiteMap's frame onto ours, through the ligand
        points = points @ rot + t
        sframe.positions = np.asarray(sframe.positions) @ rot + t
        shutil.copy(where / "apo.mae", out / "apo.mae")
        save(sframe.select('protein and not chain "LIG"').clone(), str(out / "sitemap_frame.pdb"))
        write_ligand(out / "ligand.pdb", lig)
        write_pqr(out / "fpocket_volume.pqr", centres, np.clip(radii - V.SPHERE_SHRINK, 0.05, None), "FPV")
        write_pqr(out / "fpocket_spheres.pqr", centres, radii, "FPS")
        write_pqr(out / "sitemap_volume.pqr", points, np.full(len(points), V.POINT_RADIUS), "SMV")
        lig_fit = float(np.sqrt(((slig @ rot + t - lig) ** 2).sum(1).mean()))
        (out / "view.pml").write_text(f"""# {pair}, {args.model}: our first PPc consensus pocket (rank {r['rank']} by quality x burial,
# best frame {r['best_frame']}) and SiteMap's first PPc site (rank {srow['rank']}), with the holo ligand.
# volumes measured: fpocket {float(r['pocket_volume']):.0f} A^3, SiteMap {float(srow['pocket_volume']):.0f} A^3
# DVO: fpocket {float(r['DVO']):.3f}, SiteMap {float(srow['DVO']):.3f}
# SiteMap frame moved onto ours through the ligand (ligand fit RMSD {lig_fit:.2f} A)
bg_color black
set ray_opaque_background, 0
load apo.mae, apo
load ligand.pdb, ligand
load fpocket_volume.pqr, fpocket_volume
load fpocket_spheres.pqr, fpocket_spheres
load sitemap_volume.pqr, sitemap_volume
load sitemap_frame.pdb, sitemap_frame
hide everything
show cartoon, apo
color wheat, apo
set cartoon_transparency, 0.6, apo
show sticks, ligand
color tv_blue, ligand
set stick_radius, 0.25, ligand
# our pocket: the empty part of each alpha sphere (r - 1.7 A), the volume that was measured
show spheres, fpocket_volume
color orange, fpocket_volume
set sphere_transparency, 0.55, fpocket_volume
# our pocket's full alpha spheres (they reach the bead centres), off by default
show spheres, fpocket_spheres
color yelloworange, fpocket_spheres
set sphere_transparency, 0.85, fpocket_spheres
disable fpocket_spheres
# SiteMap's site points at 0.85 A, the volume that was measured
show spheres, sitemap_volume
color cyan, sitemap_volume
set sphere_transparency, 0.4, sitemap_volume
# SiteMap's MD frame (moved onto ours), off by default
show cartoon, sitemap_frame
color palecyan, sitemap_frame
disable sitemap_frame
orient ligand
zoom ligand, 12
""")
        print(f"{out}: ours {float(r['pocket_volume']):.0f} A^3 DVO {float(r['DVO']):.3f}; SiteMap "
              f"{float(srow['pocket_volume']):.0f} A^3 DVO {float(srow['DVO']):.3f}; ligand fit {lig_fit:.2f} A")  # fmt: skip


if __name__ == "__main__":
    main()
