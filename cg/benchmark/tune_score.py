"""Tune a representation as fpocket itself was tuned: flags and pocket score on its training set.

fpocket's flags and pocket score were fitted on its training set of 263 holo
complexes (``train263``) and judged by Top-1/Top-3 under PPc (pocket center
within 4 Å of the ligand).  For each set of flags tried here, fpocket runs on
every training complex with --write_score_descriptors, the score is refitted
(score_fit.py), and the trial is judged by the mean of Top-1 and Top-3 with
the score fitted on the other folds (5-fold, complexes held out whole).

Random search over the flags, then a local search around the best.  Each
trial goes to ``runs/train263/<model>.jsonl`` with its ranks and the
coefficients fitted on every complex; the pocket tables go to
``runs/train263/<model>/<trial>.npz`` so that the score can be refitted later
without running fpocket again.

    python tune_score.py --model sirah --random 150 --local 100 -j 4
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import evaluate as E
import score_fit as F
import tune

SET = "train263"
#: around what the earlier searches found for beads (wider than all-atom)
SPACE = {**tune.SPACE, "m": (3.0, 5.5), "width": (1.5, 5.0), "M": (4.5, 10.5), "D": (1.0, 4.5),
         "e": "e"}  # fmt: skip


def objective(t: dict) -> float:
    return 0.5 * (t["cv"]["top1"] + t["cv"]["top3"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, choices=["martini2", "martini3", "sirah", "aa"])
    ap.add_argument("--random", type=int, default=150)
    ap.add_argument("--local", type=int, default=100)
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=json.loads, help="flags (JSON, as in presets.json 'params') tried before the random search, e.g. a previous preset")
    args = ap.parse_args()

    martini = args.model.startswith("martini")
    runs = tune.RUNS / SET
    (runs / args.model).mkdir(parents=True, exist_ok=True)
    log = runs / f"{args.model}.jsonl"
    done = {}
    if log.exists():
        for line in open(log):
            t = json.loads(line)
            done[tune.key(t["params"])] = t
    rng = random.Random(args.seed + len(done) + sum(map(ord, args.model)))  # stable per model
    entries = E.structures(SET, ("holo",))

    with ProcessPoolExecutor(args.jobs) as pool:

        def trial(p: dict, stage: str):
            if tune.key(p) in done:
                return done[tune.key(p)]
            flags = [*tune.flags_of(p), "--write_score_descriptors"]
            work = [(e, tune.variant_of(args.model, p), flags, 1000) for e in entries]
            results = list(pool.map(E.score_structure, work))
            structs = F.tables(results)
            coeffs = F.fit(structs)
            cv = F.cv_ranks(structs)
            name = f"{len(done):04d}"
            np.savez_compressed(runs / args.model / f"{name}.npz",
                                **{f"s{i}": (np.c_[s[0], s[1], s[2]] if s is not None else np.zeros((0, F.NF + 2)))
                                   for i, s in enumerate(structs)})  # fmt: skip
            t = {"params": p, "stage": stage, "trial": name, "cases": [r["case"] for r in results],
                 "cv": F.tops(cv), "cv_moc": F.tops(F.ranks(structs, coeffs, label=2)),
                 "in_sample": F.tops(F.ranks(structs, coeffs)),
                 "builtin_score": F.tops([r["hit_rank_dca"] for r in results]),
                 "cv_ranks": cv, "coefficients": coeffs.tolist(),
                 "mean_pockets": float(np.mean([r["npockets"] for r in results]))}  # fmt: skip
            t["objective"] = objective(t)
            done[tune.key(p)] = t
            with open(log, "a") as f:
                f.write(json.dumps(t) + "\n")
            return t

        def best():
            return max(done.values(), key=lambda t: t["objective"])

        trial(dict(tune.DEFAULT), "default")
        if args.start:
            trial({**args.start, "e": "e"}, "start")
        n_random = sum(1 for t in done.values() if t["stage"] == "random")
        for _ in range(max(0, args.random - n_random)):
            trial(tune.sample(rng, martini, SPACE), "random")
        print(f"after random search: {best()['objective']:.3f} {best()['params']}", flush=True)
        n_local = sum(1 for t in done.values() if t["stage"] == "local")
        for k in range(max(0, args.local - n_local)):
            q = tune.perturb(rng, best()["params"], martini, 1.0 if k < args.local // 2 else 0.4)
            q["e"] = "e"
            trial(q, "local")
        b = best()
        print(f"best: {b['objective']:.3f} {b['params']} cv {b['cv']}", flush=True)


if __name__ == "__main__":
    main()
