"""Presets from tune_score.py: flags and pocket score per representation, and their validation.

For each coarse-grained representation the preset is the trial of
``runs/train263/<model>.jsonl`` whose neighbourhood scores best (the mean
objective of its ``report.NEIGHBOURS`` nearest trials in parameter space),
with the score coefficients fitted on all 263 training complexes.  The
presets are then run once on sets the search never saw: fpocket's 48
apo/holo pairs (as the fpocket paper reported) and the Schrodinger apo
structures.  All-atom fpocket with its default flags and built-in score is
the reference, as are the coarse-grained structures with those defaults.

    python report_score.py      # prints the tables, writes ../presets.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import evaluate as E
import report as R
import score_fit as F
import tune

MODELS = ["martini2", "martini3", "sirah"]
VALIDATION = [("pp48", "apo"), ("pp48", "holo"), ("schrodinger", "apo")]


def load(model):
    path = tune.RUNS / "train263" / f"{model}.jsonl"
    return [json.loads(x) for x in open(path)] if path.exists() else []


def smoothed_best(trials):
    def vec(p):
        return [p[k] / R.SCALES[k] for k in R.SCALES] + [3.0 * bool(p.get("npolar"))]

    V = np.array([vec(t["params"]) for t in trials])
    obj = np.array([t["objective"] for t in trials])
    smooth = [obj[np.argsort(np.linalg.norm(V - V[i], axis=1))[: R.NEIGHBOURS]].mean()
              for i in range(len(trials))]  # fmt: skip
    return trials[int(np.argmax(smooth))]


def validate(variant, flags, jobs):
    out = {}
    for dataset, kind in VALIDATION:
        s, _ = E.evaluate(dataset, variant, flags, kinds=(kind,), jobs=jobs)
        out[f"{dataset} {kind}"] = s[kind]
    return out


def fmt(s, moc=False):
    sfx = "_moc" if moc else ""
    return " / ".join(f"{s[f'top{k}{sfx}']:.2f}" for k in (1, 3, 5, 10))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-j", "--jobs", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS,
                    help="representations to (re)make presets for; the others are left as they are")
    ap.add_argument("--pending", nargs="*", default=[], choices=MODELS,
                    help="representations whose search is not finished: their preset is withdrawn")
    args = ap.parse_args()

    path = R.HERE.parent / "presets.json"
    presets = json.loads(path.read_text()) if path.exists() else {}
    # all-atom keeps fpocket's own flags and score: it is the reference, not tuned
    presets["aa"] = {"flags": [], "n_polar": False,
                     "note": "fpocket's defaults and built-in score, unchanged"}  # fmt: skip
    for model in args.pending:
        presets[model] = {"flags": None, "pending": "its tune_score.py search is not finished"}
    rows = [("aa", "fpocket defaults", validate("aa", [], args.jobs), None)]
    for model in args.models:
        trials = load(model)
        if not trials:
            continue
        best = smoothed_best(trials)
        p = best["params"]
        variant = tune.variant_of(model, p)
        flags = [*tune.flags_of(p), F.coefficient_flag(best["coefficients"])]
        rows.append((model, "fpocket defaults", validate(model, [], args.jobs), None))
        tuned = validate(variant, flags, args.jobs)
        rows.append((model, "tuned flags + score", tuned, best))
        old = presets.get(model, {})
        presets[model] = {"flags": flags, "n_polar": bool(p.get("npolar", False)), "params": p,
                          "score_coefficients": best["coefficients"],
                          "score_features": list(E.SCORE_FEATURES),
                          "trained_on": "train263", "trials": len(trials),
                          "train_cv": best["cv"], "train_cv_moc": best["cv_moc"],
                          "validation": tuned}  # fmt: skip
        if "mdpocket_density_iso" in old:
            presets[model]["mdpocket_density_iso"] = old["mdpocket_density_iso"]

    print("PPc (pocket center < 4 Å from a ligand atom): Top-1 / Top-3 / Top-5 / Top-10\n")
    header = "| model | setting | " + " | ".join(f"{d} {k}" for d, k in VALIDATION) + " |"
    for moc in (False, True):
        if moc:
            print("\nMOc (mutual overlap): Top-1 / Top-3 / Top-5 / Top-10\n")
        print(header)
        print("|---|---|" + "---|" * len(VALIDATION))
        for model, setting, v, _ in rows:
            print(f"| {model} | {setting} | " + " | ".join(fmt(v[f'{d} {k}'], moc) for d, k in VALIDATION) + " |")
    print("\ntraining set (263 holo complexes), score fitted on the other folds, PPc T1/T3/T5/T10:")
    for model, setting, _, best in rows:
        if best:
            c = best["cv"]
            print(f"  {model:9s} {c['top1']:.2f} / {c['top3']:.2f} / {c['top5']:.2f} / {c['top10']:.2f}"
                  f"   {' '.join(tune.flags_of(best['params']))}"
                  f"{'  (N beads polar)' if best['params'].get('npolar') else ''}")  # fmt: skip
    path.write_text(json.dumps(presets, indent=1))


if __name__ == "__main__":
    main()
