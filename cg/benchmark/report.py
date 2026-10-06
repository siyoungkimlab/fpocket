"""Summarize a search: best flags on the training set, cross-validation, test set, presets.json.

Training set (``--train``, default pp48): the trials in ``runs/<train>/``.
The preset of each representation is the trial whose neighbourhood scores
best: each trial is scored by the mean objective of its ``NEIGHBOURS``
nearest trials in parameter space (itself included).  With ~100 structures
the single best trial is partly luck; a broad optimum carries over to new
proteins better than a narrow spike.

Cross-validation reuses those trials: structures are split into folds by
protein (a pair's apo and holo structures stay together), the best trial on
the training folds is picked, and its hit ranks on the held-out fold are
counted.  Only the default and random-search trials take part: the local
search was centered on the optimum over every structure, which would leak.

Test set (``--test``, default schrodinger): the preset, and fpocket's
default flags, run once on a set the search never saw.

``--target apo`` (the default) is the task: fpocket runs on the apo
structure, and the holo structure only says where the site is.  ``--target
all`` also counts fpocket runs on the holo structures.

    python report.py                     # prints the tables, writes ../presets.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

import evaluate as E
import tune

HERE = Path(__file__).resolve().parent
MODELS = ["aa", "martini2", "martini3", "sirah"]
FOLDS, REPEATS = 5, 20
NEIGHBOURS = 8
#: parameter steps of similar consequence, for the neighbourhood distance
SCALES = {"m": 0.3, "M": 0.6, "D": 0.4, "i": 5, "A": 1.5}


def load(dataset, model):
    path = HERE / "runs" / dataset / f"{model}.jsonl"
    return [json.loads(x) for x in open(path)] if path.exists() else []


def objective(ranks, keys):
    t = tune.tops(ranks, keys)
    return (t["top1"] + t["top3"] + t["top5"]) / 3


def summary_of(trial, kind):
    """Top-1/3/5 of a stored trial on one kind of structure, by the ligand criterion."""
    return tune.tops(trial["ranks_dca"], tune.target_keys(trial, kind))


def cross_validate(trials, target, seed=0):
    keys = sorted(tune.target_keys(trials[0], target))
    groups = sorted({k.split(":", 1)[1] for k in keys})
    rng = random.Random(seed)
    held = defaultdict(list)
    for _ in range(REPEATS):
        order = groups[:]
        rng.shuffle(order)
        for f in range(FOLDS):
            test_groups = set(order[f::FOLDS])
            test = [k for k in keys if k.split(":", 1)[1] in test_groups]
            train = [k for k in keys if k.split(":", 1)[1] not in test_groups]
            best = max(trials, key=lambda t: objective(t["ranks_dca"], train))
            for k in test:
                held[k].append(best["ranks_dca"][k])
    out = {}
    for name, prefix in (("apo", "apo:"), ("holo", "holo:"), ("all", "")):
        ks = [k for k in keys if k.startswith(prefix)]
        if not ks:
            continue
        out[name] = {f"top{n}": float(np.mean([np.mean([r is not None and r <= n for r in held[k]])
                                               for k in ks])) for n in (1, 3, 5)}  # fmt: skip
    return out


def best_trial(trials, target):
    """The trial with the best mean objective over its nearest neighbours."""
    def vec(p):
        return [p[k] / SCALES[k] for k in SCALES] + [3.0 * bool(p.get("npolar")), 3.0 * (p["e"] == "b")]

    V = np.array([vec(t["params"]) for t in trials])
    obj = np.array([tune.objective(t, target) for t in trials])
    smooth = [obj[np.argsort(np.linalg.norm(V - V[i], axis=1))[:NEIGHBOURS]].mean()
              for i in range(len(trials))]  # fmt: skip
    return trials[int(np.argmax(smooth))]


def on_test(dataset, model, params, kinds):
    summary, _ = E.evaluate(dataset, tune.variant_of(model, params), tune.flags_of(params),
                            kinds=kinds, jobs=10)  # fmt: skip
    return summary


def fmt(s):
    return f"{s['top1']:.2f} / {s['top3']:.2f} / {s['top5']:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--train", default="pp48", choices=E.SETS)
    ap.add_argument("--test", default="schrodinger", choices=E.SETS)
    ap.add_argument("--no-test", action="store_true", help="skip the test-set runs")
    ap.add_argument("--target", default="apo", choices=["apo", "all"])
    args = ap.parse_args()
    target = args.target
    kinds = ("apo",) if target == "apo" else ("apo", "holo")

    presets, rows = {}, []
    for model in MODELS:
        trials = [t for t in load(args.train, model) if target in t["summary"] and "ranks_dca" in t]
        if not trials:
            continue
        default = next(t for t in trials if t["stage"] == "default")
        best = best_trial(trials, target)
        cv = cross_validate([t for t in trials if t["stage"] in ("default", "random")], target)
        test = test_default = None
        if not args.no_test:
            test = on_test(args.test, model, best["params"], kinds)
            test_default = on_test(args.test, model, tune.DEFAULT, kinds)
        p = best["params"]
        presets[model] = {"flags": tune.flags_of(p), "n_polar": bool(p.get("npolar", False)),
                          "params": p, "trained_on": args.train, "target": target,
                          "trials": len(trials),
                          "train": {k: summary_of(best, k) for k in kinds}, "cv": cv,
                          "test_set": args.test, "test": test, "test_default_flags": test_default}  # fmt: skip
        rows.append((model, default, best, cv, test, test_default, len(trials)))

    print("Top-1 / Top-3 / Top-5: fraction of structures with a pocket whose center is within 4 Å\n"
          "of a holo ligand atom among the first k pockets\n")
    for kind in kinds:
        print(f"{kind}:\n")
        print(f"| model | trials | {args.train} default | {args.train} tuned | {args.train} CV "
              f"| {args.test} default | {args.test} tuned |")  # fmt: skip
        print("|---|---|---|---|---|---|---|")
        for model, default, best, cv, test, test_default, n in rows:
            t = fmt(test[kind]) if test else "-"
            td = fmt(test_default[kind]) if test_default else "-"
            print(f"| {model} | {n} | {fmt(summary_of(default, kind))} | {fmt(summary_of(best, kind))} "
                  f"| {fmt(cv[kind])} | {td} | {t} |")  # fmt: skip
        print()
    for model, _, best, *_ in rows:
        print(f"{model:9s} {' '.join(tune.flags_of(best['params']))}"
              f"{'   (Martini N beads polar)' if best['params'].get('npolar') else ''}")  # fmt: skip
    (HERE.parent / "presets.json").write_text(json.dumps(presets, indent=1))


if __name__ == "__main__":
    main()
