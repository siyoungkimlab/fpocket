"""Refit a SiteScore-like score on fpocket's training set (an experiment).

For every fpocket pocket (model preset) of the 263 training complexes, its
enclosed core (core.py) on the static structure gives

  V  core volume (Å^3)        e  enclosure: share of 60 rays from each core point that
                                 meet a bead within 10 Å, averaged over the core
  s  fpocket's own score (the preset's refitted coefficients)

and the label PPc_core (core centre < 4 Å from a ligand atom). Two scores are fitted
by grid search to maximise Top-1, 5-fold cross-validated by complex:

  A  sqrt(min(V, cap)) + w * e                    SiteScore's form (Halgren 2009)
  B  s + a * sqrt(min(V, cap)) + b * e            the current score plus SiteMap's terms

and checked on fpocket's 48 apo structures. Bead radii: cg/radii/ by bead type
(Martini types from bead_types.json, SIRAH types from its residue library).

    python sitescore_fit.py --models martini2 martini3 sirah -j 12
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import cgpocket as C  # noqa: E402
import cgprep  # noqa: E402
import core as K  # noqa: E402
import evaluate as E  # noqa: E402

OUT = HERE / "runs" / "sitescore"
CAPS = np.arange(50, 1001, 50)


def bead_types(names, resnames, model):
    if model.startswith("martini"):
        table = json.loads((HERE.parent / "bead_types.json").read_text())[model]
        return [table.get(r, {}).get(n, "") for n, r in zip(names, resnames, strict=True)]
    from boonza.sirah.build import read_residues

    lib = read_residues()[0]
    std2sirah = {}
    for s, std in cgprep._SIRAH_RESNAMES.items():
        std2sirah.setdefault(std, s)
    out = []
    for n, r in zip(names, resnames, strict=True):
        entry = lib.get(std2sirah.get(r, r))
        kinds = {a: k for a, k, _ in entry.atoms} if entry else {}
        out.append(kinds.get(n, ""))
    return out


def read_beads(pdb: Path):
    xyz, names, res = [], [], []
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")):
            names.append(line[12:16].strip())
            res.append(line[17:21].strip())
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(xyz), names, res


def enclosure(points, beads, radii, n_rays=60, ray=10.0):
    tree = cKDTree(beads)
    dirs = K._directions(n_rays)
    steps = np.arange(1.0, ray + 0.5, 1.0)
    probes = (points[:, None, None, :] + dirs[None, :, None, :] * steps[None, None, :, None]).reshape(-1, 3)
    d, j = tree.query(probes, k=4)
    return float((d <= radii[j]).any(1).reshape(len(points), n_rays, len(steps)).any(2).mean())


def one(args):
    entry, model, variant, flags, coeffs = args
    pdb = E.DATA / entry["set"] / "pdb" / f"{entry['stem']}_{variant}.pdb"
    beads, names, res = read_beads(pdb)
    radii = K.bead_radii(bead_types(names, res, model), model)
    radii[np.isnan(radii)] = np.nanmedian(radii) if np.isfinite(radii).any() else 2.0
    lig = np.array(entry["ligand"])
    rows = []
    for centres, sradii, feats in E.run_fpocket(pdb, flags):
        s = float(coeffs[0] + np.dot(coeffs[1:], feats)) if feats else np.nan
        pts = K.core_points(centres, sradii, beads, radii)
        if len(pts):
            v = len(pts) * K.SPACING**3
            e = enclosure(pts, beads, radii)
            ppc = float(np.linalg.norm(lig - pts.mean(0), axis=1).min()) < C.PPC_CUTOFF
        else:
            v, e, ppc = 0.0, 0.0, False
        rows.append((v, e, s, ppc))
    return entry["case"] + ":" + entry["stem"], rows


def arrays(data):
    return {k: np.nan_to_num(np.array(v, float).reshape(-1, 4), nan=-1e9) for k, v in data.items()}


def table(dataset, kind, model, jobs):
    path = OUT / f"{dataset}_{kind}_{model}.json"
    if path.exists():
        return arrays(json.loads(path.read_text()))
    preset = json.loads(C.PRESETS.read_text())[model]
    variant = f"{model}_npolar" if preset.get("n_polar") else model
    flags = [*preset["flags"], "--write_score_descriptors"]
    jobs_ = [(e, model, variant, flags, preset["score_coefficients"]) for e in E.structures(dataset, (kind,))]
    with ProcessPoolExecutor(jobs) as pool:
        data = dict(pool.map(one, jobs_))
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return arrays(data)


def ranks(data, score):
    """First rank of a PPc_core pocket per structure under ``score(V, e, s)``."""
    out = []
    for a in data.values():
        if not len(a):
            out.append(None)
            continue
        sc = score(a[:, 0], a[:, 1], a[:, 2])
        order = np.argsort(-sc, kind="stable")
        hit = np.flatnonzero(a[order, 3] > 0)
        out.append(int(hit[0]) + 1 if len(hit) else None)
    return out


def top1(r):
    return sum(1 for x in r if x == 1) / len(r)


def tops(r):
    return " / ".join(f"{sum(1 for x in r if x and x <= k) / len(r):.2f}" for k in (1, 3, 5, 10))


def form_a(cap, w):
    return lambda V, e, s: np.sqrt(np.minimum(V, cap)) + w * e


def form_b(cap, a, b):
    return lambda V, e, s: s + a * np.sqrt(np.minimum(V, cap)) + b * e


GRID_A = [(cap, w) for cap in CAPS for w in np.r_[0, np.geomspace(0.5, 300, 25)]]
GRID_B = [(cap, a, b) for cap in CAPS[::2] for a in np.r_[0, np.geomspace(0.01, 2, 10)]
          for b in np.r_[0, np.geomspace(0.1, 20, 10)]]  # fmt: skip


def fit(data, grid, form):
    """The grid point with the best Top-1, then the best Top-3."""
    def key(g):
        r = ranks(data, form(*g))
        return top1(r), sum(1 for x in r if x and x <= 3)
    return max(grid, key=key)


def cv(data, grid, form, k=5, seed=0):
    keys = sorted(data)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(keys)), k)
    out = {}
    for f in folds:
        test = {keys[i] for i in f}
        train = {x: data[x] for x in keys if x not in test}
        g = fit(train, grid, form)
        out.update(dict(zip(sorted(test), ranks({x: data[x] for x in sorted(test)}, form(*g)), strict=True)))
    return [out[x] for x in keys]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["martini2", "martini3", "sirah"])
    ap.add_argument("-j", "--jobs", type=int, default=12)
    args = ap.parse_args()
    fitted = {}
    for model in args.models:
        train = table("train263", "holo", model, args.jobs)
        check = table("pp48", "apo", model, args.jobs)
        base = lambda V, e, s: s  # noqa: E731
        ga, gb = fit(train, GRID_A, form_a), fit(train, GRID_B, form_b)
        fitted[model] = {"A": [float(x) for x in ga], "B": [float(x) for x in gb]}
        print(f"\n== {model}: {len(train)} training complexes, {len(check)} apo checks; PPc from the core centre")
        print(f"  fpocket score (current)   train CV {tops(ranks(train, base))}   48 apo {tops(ranks(check, base))}")
        print(f"  A sqrt(min(V,{ga[0]:.0f})) + {ga[1]:.2f} e   train CV {tops(cv(train, GRID_A, form_a))}   "
              f"48 apo {tops(ranks(check, form_a(*ga)))}")
        print(f"  B s + {gb[1]:.3f} sqrt(min(V,{gb[0]:.0f})) + {gb[2]:.2f} e   train CV {tops(cv(train, GRID_B, form_b))}   "
              f"48 apo {tops(ranks(check, form_b(*gb)))}")
    (OUT / "fitted.json").write_text(json.dumps(fitted, indent=1))


if __name__ == "__main__":
    main()
