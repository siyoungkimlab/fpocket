"""The mdpocket density isovalue for each representation, written into ../presets.json.

mdpocket's density map counts alpha-sphere centers near each grid point
(``*_dens.dx``) and contours it at 8 (``*_dens_iso_8.pdb``), a level set for
all-atom structures with fpocket's default flags.  Beads give fewer spheres
per pocket, so the same level cuts CG pockets away.  Each holo structure of
the set is run through mdpocket as a one-frame trajectory with its preset;
the peak density within 3 Å of the ligand is compared with all-atom at the
default flags, and 8 is scaled by the ratio of the medians.

The frequency map (``*_freq_iso_0_5.pdb``: the fraction of frames with a
pocket there) needs no such scaling.

    python calibrate_density.py --set pp48
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import evaluate as E

MDPOCKET = E.HERE.parents[1] / "bin" / "mdpocket"
PRESETS = E.HERE.parent / "presets.json"
ALL_ATOM_ISO = 8.0
NEAR = 3.0  # Å from a ligand atom


def read_dx(path):
    values, counts, origin = [], None, None
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith("#"):
            continue
        if line.startswith("object 1"):
            counts = tuple(int(x) for x in w[-3:])
        elif w[0] == "origin":
            origin = np.array(w[1:4], float)
        elif w[0][0].isdigit() or w[0][0] in "-.":
            values += [float(x) for x in w]
    return np.array(values[: int(np.prod(counts))]).reshape(counts), origin


def peak_density(args):
    entry, variant, flags = args
    pdb = E.DATA / entry["set"] / "pdb" / f"{entry['stem']}_{variant}.pdb"
    with tempfile.TemporaryDirectory(dir=os.environ.get("FPOCKET_TMP")) as tmp:
        (Path(tmp) / "frames.txt").write_text(f"{pdb}\n")
        subprocess.run([str(MDPOCKET), "-L", f"{tmp}/frames.txt", "-o", f"{tmp}/md", *flags],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)  # fmt: skip
        dx = Path(f"{tmp}/md_dens.dx")
        if not dx.exists():
            return None
        grid, origin = read_dx(dx)
    idx = np.argwhere(grid > 0)
    xyz = origin + idx  # mdpocket's grid is 1 Å
    near = np.linalg.norm(xyz[:, None] - np.array(entry["ligand"])[None], axis=2).min(1) < NEAR
    return float(grid[tuple(idx[near].T)].max()) if near.any() else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--set", default="pp48", choices=E.SETS)
    ap.add_argument("-j", "--jobs", type=int, default=10)
    args = ap.parse_args()
    presets = json.loads(PRESETS.read_text())
    entries = E.structures(args.set, ("holo",))

    def median(pool, variant, flags):
        v = [x for x in pool.map(peak_density, [(e, variant, flags) for e in entries]) if x is not None]
        return float(np.median(v)), len(v)

    with ProcessPoolExecutor(args.jobs) as pool:
        reference, n = median(pool, "aa", [])
        print(f"all-atom, default flags: median peak site density {reference:.1f} (n={n})")
        for model, p in presets.items():
            if p.get("flags") is None:
                continue  # a preset still being made
            variant = f"{model}_npolar" if p["n_polar"] else model
            peak, n = median(pool, variant, p["flags"])
            iso = round(2 * ALL_ATOM_ISO * peak / reference) / 2  # to the nearest 0.5
            p["mdpocket_density_iso"] = iso
            print(f"{model:9s} median peak {peak:.1f} (n={n}): density isovalue {iso}")
    PRESETS.write_text(json.dumps(presets, indent=1))


if __name__ == "__main__":
    main()
