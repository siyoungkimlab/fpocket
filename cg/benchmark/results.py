"""Reading a ``boonza pockets`` output directory, or one written by the cgpocket.py it replaced.

The trajectory runs of 2026-10 were written by ``cg/cgpocket.py traj --rank``:
pockets_vs_holo.csv, the beads in mdpocket/md.{pdb,dcd}, columns ``burial`` and
``volume_best_frame``.  ``boonza pockets traj`` writes pockets.csv, md.{pdb,dcd} at
the top, ``pocket_burial`` and ``fpocket_volume``.  The analyses here read either.
"""

from __future__ import annotations

import csv
from pathlib import Path

#: old column name -> new
RENAMED = {"burial": "pocket_burial", "volume_best_frame": "fpocket_volume",
           "alpha_spheres_best_frame": "alpha_spheres"}  # fmt: skip


def table(where: Path) -> list[dict]:
    """The consensus pockets of one run, with the new column names."""
    where = Path(where)
    path = where / "pockets.csv"
    if not path.exists():
        path = where / "pockets_vs_holo.csv"
    return [{RENAMED.get(k, k): v for k, v in r.items()} for r in csv.DictReader(open(path))]


def has_table(where: Path) -> bool:
    return (Path(where) / "pockets.csv").exists() or (Path(where) / "pockets_vs_holo.csv").exists()


def md(where: Path) -> Path:
    """The fitted protein-bead trajectory (its .pdb is beside it)."""
    where = Path(where)
    return where / "md.dcd" if (where / "md.dcd").exists() else where / "mdpocket" / "md.dcd"


def ligand_on(holo, reference, ligand: str = "resname LIG", with_fit: bool = False):
    """The holo ligand's heavy atoms in ``reference``'s frame: the whole holo protein (alpha
    carbons, every chain) superposed by sequence on the reference's backbone (CA, BB or GC),
    the ligand moved with it -- the placement every benchmark number here was made with.
    ``with_fit``: also the boonza.superpose result."""
    import boonza
    import numpy as np

    names = set(np.asarray(reference.atoms["name"]).tolist())
    back = next(n for n in ("CA", "BB", "GC") if n in names)
    moved = holo.clone()
    fit = boonza.superpose(moved, reference, sel='protein and name CA and not chain "LIG"',
                     ref_sel=f'name {back} and not chain "LIG"', match="sequence", apply=True)  # fmt: skip
    lig = np.asarray(moved.positions)[moved.select(f"({ligand}) and not element H").ids]
    return (lig, fit) if with_fit else lig
