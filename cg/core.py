"""The enclosed core of an fpocket pocket, after SiteMap's definition of site points.

fpocket's pocket is all the empty space its alpha spheres describe, including
the solvent-exposed fringe; SiteMap keeps only grid points that are outside the
protein, in contact with it and enclosed by it (Halgren 2009). Here a pocket is
trimmed to such points on one frame's beads, each bead with its own radius R_i
(boonza.sites.particle_radii, sigma/2 from the force field):

  inside the pocket   within one of the pocket's alpha spheres
  outside the protein distance to every bead >= OUTSIDE * R_i  (SiteMap: d^2 >= 2.5 r^2)
  in contact          some bead within R_i + CONTACT Å        (stands in for SiteMap's
                                                               vdW-contact test)
  enclosed            >= ENCLOSURE of N_RAYS rays meet a bead (within its R_i) inside
                      RAY_LENGTH Å                            (SiteMap: 0.5 within 8 Å)
  not isolated        >= MIN_NEIGHBOURS kept neighbours among the 18 face/edge
                      neighbours                              (SiteMap: >= 3 within 1.76 Å)

on a SPACING grid. Each kept point stands for SPACING^3 of volume.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SPACING = 2.0
OUTSIDE = np.sqrt(2.5)
CONTACT = 3.0
ENCLOSURE = 0.5
RAY_LENGTH = 8.0
N_RAYS = 60
MIN_NEIGHBOURS = 2


def _directions(n: int) -> np.ndarray:
    """n near-uniform unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


def core_points(centres, radii, beads, bead_radii, spacing: float = SPACING) -> np.ndarray:
    """Grid points (n, 3) of the enclosed core of one pocket (its alpha spheres) on one
    frame (``beads`` with per-bead ``bead_radii``)."""
    centres, radii, beads = (np.asarray(x, float) for x in (centres, radii, beads))
    bead_radii = np.asarray(bead_radii, float)
    lo, hi = (centres - radii[:, None]).min(0), (centres + radii[:, None]).max(0)
    axes = [np.arange(np.floor(a / spacing), np.ceil(b / spacing) + 1) * spacing for a, b in zip(lo, hi, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, j = cKDTree(centres).query(grid, k=min(8, len(centres)))
    d, j = d.reshape(len(grid), -1), j.reshape(len(grid), -1)
    pts = grid[(d <= radii[j]).any(1)]
    if not len(pts):
        return pts
    tree = cKDTree(beads)
    k = min(12, len(beads))
    dd, jj = tree.query(pts, k=k)
    outside = (dd >= OUTSIDE * bead_radii[jj]).all(1)
    contact = (dd <= bead_radii[jj] + CONTACT).any(1)
    pts = pts[outside & contact]
    if not len(pts):
        return pts
    dirs = _directions(N_RAYS)
    steps = np.arange(1.0, RAY_LENGTH + 0.5, 1.0)
    probes = (pts[:, None, None, :] + dirs[None, :, None, :] * steps[None, None, :, None]).reshape(-1, 3)
    dd, jj = tree.query(probes, k=min(4, len(beads)))
    hit = (dd <= bead_radii[jj]).any(1).reshape(len(pts), N_RAYS, len(steps)).any(2).mean(1)
    pts = pts[hit >= ENCLOSURE]
    if len(pts) < 2:
        return pts
    nb = cKDTree(pts).query_ball_point(pts, spacing * np.sqrt(2) + 1e-6)
    return pts[np.array([len(x) - 1 for x in nb]) >= MIN_NEIGHBOURS]


def cell_radius(spacing: float = SPACING) -> float:
    """Radius of a sphere with the volume of one grid cell (for volume.py voxels)."""
    return spacing * (3 / (4 * np.pi)) ** (1 / 3)
