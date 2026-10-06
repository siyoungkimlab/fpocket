"""Tune fpocket's pocket-detection flags for one representation.

Random search over the flags, then a local search around the best (see
evaluate.py for the scoring).  The task is ``--target apo``: fpocket runs on
the apo structure, and the holo structure only says where the site is.
(``--target all`` also counts runs on the holo structures, as the first
searches did.)  The objective is the mean of Top-1, Top-3 and Top-5; Top-5
keeps a preset from scoring well by returning almost no pockets.  A pocket
is correct when its center is within 4 Å of a holo ligand atom (evaluate.py);
the objective is computed from each trial's stored ranks, so old trials are
rescored when the criterion or the target changes.  Every trial is appended to
``runs/<set>/<model>.jsonl`` with each structure's hit rank, so that the search
can be resumed and cross-validated without running fpocket again
(``report.py``).

    python tune.py --set pp48 --model martini3 --random 300 --local 150 -j 3

Trials are shared between targets: one run with apo and holo results serves
both, and is rescored for the target when the log is read.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import evaluate as E

RUNS = E.HERE / "runs"

#: fpocket's defaults, the first trial of every search
DEFAULT = {"m": 3.4, "M": 6.2, "D": 2.4, "i": 15, "A": 3, "C": "s", "e": "e", "npolar": False}
SPACE = {
    "m": (2.8, 6.5),
    "width": (1.0, 5.0),  # M - m
    "M": (4.0, 11.0),
    "D": (1.0, 6.0),
    "i": (2, 40),
    "A": (0, 4),
    "C": "s",  # other linkages are ~10x slower: too slow for trajectories
    "e": "eb",
}


def flags_of(p: dict) -> list[str]:
    return ["-m", f"{p['m']:.2f}", "-M", f"{p['M']:.2f}", "-D", f"{p['D']:.2f}",
            "-i", str(p["i"]), "-A", str(p["A"]), "-C", p["C"], "-e", p["e"]]  # fmt: skip


def variant_of(model: str, p: dict) -> str:
    return f"{model}_npolar" if p.get("npolar") else model


#: SIRAH beads sit on real atoms (N, CA, O and one side-chain atom), so a SIRAH
#: protein is a thinned all-atom one: search it, and all-atom, around fpocket's
#: all-atom defaults rather than over the Martini range
NEAR_ATOMIC = {"m": (3.0, 5.0), "width": (1.5, 4.5), "M": (4.5, 9.5), "D": (1.5, 4.5)}
SPACES = {"aa": {**SPACE, **NEAR_ATOMIC}, "sirah": {**SPACE, **NEAR_ATOMIC},
          "martini2": SPACE, "martini3": SPACE}  # fmt: skip


def tops(ranks: dict, keys) -> dict:
    """Top-1/3/5 over ``keys`` of a {structure: rank of the first correct pocket} dict."""
    keys = list(keys)
    return {f"top{k}": sum(1 for s in keys if ranks[s] is not None and ranks[s] <= k) / len(keys)
            for k in (1, 3, 5)}  # fmt: skip


def target_keys(trial: dict, target: str) -> list:
    return [k for k in trial["ranks_dca"] if target == "all" or k.startswith(f"{target}:")]


def objective(trial: dict, target: str) -> float:
    t = tops(trial["ranks_dca"], target_keys(trial, target))
    return (t["top1"] + t["top3"] + t["top5"]) / 3


def sample(rng: random.Random, martini: bool, space: dict) -> dict:
    m = round(rng.uniform(*space["m"]), 2)
    return {
        "m": m,
        "M": round(min(m + rng.uniform(*space["width"]), space["M"][1]), 2),
        "D": round(rng.uniform(*space["D"]), 2),
        "i": rng.randint(*space["i"]),
        "A": rng.randint(*space["A"]),
        "C": rng.choice(space["C"]),
        "e": rng.choice(space["e"]),
        "npolar": rng.random() < 0.5 if martini else False,
    }


def perturb(rng: random.Random, p: dict, martini: bool, scale: float) -> dict:
    q = dict(p)
    for _ in range(rng.randint(1, 3)):
        k = rng.choice(["m", "M", "D", "i", "A", "C", "e"] + (["npolar"] if martini else []))
        if k in ("m", "M", "D"):
            q[k] = round(q[k] + rng.gauss(0, scale * (1.0 if k != "D" else 0.8)), 2)
        elif k == "i":
            q[k] = q[k] + rng.choice([-1, 1]) * rng.randint(1, max(1, int(6 * scale)))
        elif k == "A":
            q[k] = q[k] + rng.choice([-1, 1])
        elif k == "C":
            q[k] = rng.choice(SPACE["C"])
        elif k == "e":
            q[k] = rng.choice(SPACE["e"])
        else:
            q[k] = not q[k]
    q["m"] = min(max(q["m"], SPACE["m"][0]), SPACE["m"][1])
    q["M"] = min(max(q["M"], q["m"] + 0.3), SPACE["M"][1])
    q["D"] = min(max(q["D"], 0.5), 10.0)
    q["i"] = min(max(q["i"], 2), 60)  # a one-sphere pocket has no density (NaN)
    q["A"] = min(max(q["A"], 0), 4)
    return q


def key(p: dict) -> str:
    return json.dumps(p, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--set", default="pp48", choices=E.SETS)
    ap.add_argument("--model", required=True, choices=["aa", "martini2", "martini3", "sirah"])
    ap.add_argument("--random", type=int, default=300)
    ap.add_argument("--local", type=int, default=150)
    ap.add_argument("-j", "--jobs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="apo", choices=["apo", "all"],
                    help="structures fpocket is judged on (default: apo)")
    args = ap.parse_args()
    target = args.target
    local_stage = "local_dca" if target == "all" else f"local_dca_{target}"

    martini = args.model.startswith("martini")
    runs = RUNS / args.set
    runs.mkdir(parents=True, exist_ok=True)
    log = runs / f"{args.model}.jsonl"
    done = {}
    if log.exists():
        for line in open(log):
            t = json.loads(line)
            if "ranks_dca" not in t or target not in t["summary"]:
                continue  # a trial that did not run on these structures
            t["objective"] = objective(t, target)  # rescored for this target and criterion
            done[key(t["params"])] = t
    rng = random.Random(args.seed + len(done))
    kinds = ("apo", "holo") if target == "all" else (target,)
    entries = E.structures(args.set, kinds)

    with ProcessPoolExecutor(args.jobs) as pool:

        def trial(p: dict, stage: str):
            if key(p) in done:
                return done[key(p)]
            work = [(e, variant_of(args.model, p), flags_of(p), 10) for e in entries]
            results = list(pool.map(E.score_structure, work))
            summary = {k: E.summarize([r for r in results if r["kind"] == k]) for k in kinds}
            if target == "all":
                summary["all"] = E.summarize(results)
            t = {"params": p, "stage": stage, "summary": summary,
                 "ranks": {f"{r['kind']}:{r['case']}": r["hit_rank"] for r in results},
                 "ranks_dca": {f"{r['kind']}:{r['case']}": r["hit_rank_dca"] for r in results},
                 "npockets": {f"{r['kind']}:{r['case']}": r["npockets"] for r in results}}  # fmt: skip
            t["objective"] = objective(t, target)
            done[key(p)] = t
            with open(log, "a") as f:
                f.write(json.dumps(t) + "\n")
            return t

        def best():
            return max(done.values(), key=lambda t: t["objective"])

        trial(dict(DEFAULT), "default")
        n_random = sum(1 for t in done.values() if t["stage"] == "random")
        for _ in range(max(0, args.random - n_random)):
            trial(sample(rng, martini, SPACES[args.model]), "random")
        print(f"after random search: {best()['objective']:.3f} {best()['params']}", flush=True)

        n_local = sum(1 for t in done.values() if t["stage"] == local_stage)
        for k in range(max(0, args.local - n_local)):
            scale = 1.0 if k < args.local // 2 else 0.4
            trial(perturb(rng, best()["params"], martini, scale), local_stage)
        b = best()
        print(f"best: {b['objective']:.3f} {b['params']}\n{json.dumps(b['summary'])}", flush=True)


if __name__ == "__main__":
    main()
