"""Pocket-ligand overlap as volumes on a common grid, independent of how densely a pocket is described.

Point-based overlap (share of ligand atoms within 3 Å of a pocket point) favours
dense descriptions: SiteMap's site points fill a site on a 1 Å grid, fpocket's
alpha-sphere centres are ~10x sparser and clumped. Here every pocket and the
ligand become sets of voxels on the same lattice (voxel centres every SPACING Å):

- fpocket pocket: the empty part of each alpha sphere, a sphere of radius r - R
  around its centre (r reaches the centres of the atoms or beads it touches; R is
  their radius, fpocket's 1.7 Å for carbon by default), unioned. No protein
  coordinates are needed.
- point-set pocket (e.g. SiteMap site points, a 1 Å grid): voxels within
  POINT_RADIUS of a point, chosen so the voxel volume matches SiteMap's own volume
  (as the alpha-sphere voxels match fpocket's own pocket volume).
- ligand: voxels within LIGAND_RADIUS of a heavy atom; the "shell" adds SHELL Å.

Measures (fractions of voxel counts):
  ligand_volume_covered   |L & P| / |L|
  pocket_volume_in_ligand |L & P| / |P|
  pocket_volume_near_ligand |shell(L) & P| / |P|
  DVO                     |L & P| / |L | P|   (discretized volume overlap)
"""

from __future__ import annotations

import numpy as np

SPACING = 0.5  # Å; 1 Å is too coarse to match a pocket's own reported volume
SPHERE_SHRINK = 1.7  # Å: fpocket's carbon radius
LIGAND_RADIUS = 1.7  # Å: heavy atoms
SHELL = 2.0  # Å around the ligand for pocket_volume_near_ligand


def _ball(radius: float) -> np.ndarray:
    n = int(np.ceil(radius / SPACING))
    g = np.mgrid[-n : n + 1, -n : n + 1, -n : n + 1].reshape(3, -1).T
    return g[(g * SPACING) ** 2 @ np.ones(3) <= radius**2]


def _voxels_within(centres: np.ndarray, radii) -> np.ndarray:
    """Integer voxels whose centres lie within ``radii`` of ``centres``."""
    centres = np.asarray(centres, float).reshape(-1, 3)
    radii = np.broadcast_to(np.asarray(radii, float), (len(centres),))
    out = []
    for c, r in zip(centres, radii, strict=True):
        if r <= 0:
            continue
        base = np.round(c / SPACING).astype(int)
        cand = base + _ball(r + SPACING)
        keep = np.linalg.norm(cand * SPACING - c, axis=1) <= r
        out.append(cand[keep])
    return np.unique(np.vstack(out), axis=0) if out else np.zeros((0, 3), int)


def sphere_pocket(centres, radii, shrink: float = SPHERE_SHRINK) -> np.ndarray:
    """Voxels of the empty space of alpha spheres (radius r - shrink)."""
    return _voxels_within(centres, np.asarray(radii, float) - shrink)


#: Å around each SiteMap site point so the voxel volume matches SiteMap's own
#: r_sitemap_volume (calibrated on 48 sites; see point_pocket)
POINT_RADIUS = 0.85


def point_pocket(points, radius: float = POINT_RADIUS) -> np.ndarray:
    """Voxels within ``radius`` of any point (site points are a 1 Å grid; each stands for ~1 Å^3)."""
    return _voxels_within(points, radius)


def ligand_voxels(atoms, extra: float = 0.0) -> np.ndarray:
    return _voxels_within(atoms, LIGAND_RADIUS + extra)


def _count(a: np.ndarray, b: np.ndarray) -> int:
    if not len(a) or not len(b):
        return 0
    row = [("", a.dtype)] * 3
    return len(np.intersect1d(a.view(row).ravel(), b.astype(a.dtype).view(row).ravel()))


def overlap(pocket: np.ndarray, ligand_atoms) -> dict:
    """Volume-overlap measures of one pocket (voxels) with one ligand (heavy-atom coordinates)."""
    lig = ligand_voxels(ligand_atoms)
    shell = ligand_voxels(ligand_atoms, SHELL)
    pocket = np.ascontiguousarray(pocket, dtype=int)
    lig, shell = np.ascontiguousarray(lig), np.ascontiguousarray(shell)
    inter = _count(lig, pocket)
    near = _count(shell, pocket)
    union = len(lig) + len(pocket) - inter
    return {"pocket_volume": len(pocket) * SPACING**3,
            "ligand_volume_covered": inter / len(lig) if len(lig) else 0.0,
            "pocket_volume_in_ligand": inter / len(pocket) if len(pocket) else 0.0,
            "pocket_volume_near_ligand": near / len(pocket) if len(pocket) else 0.0,
            "DVO": inter / union if union else 0.0}  # fmt: skip
