"""Write ../bead_types.json: the Martini bead type of each (residue, bead name).

A Martini structure read without its topology (a bare .gro or .pdb) has bead
names but no types, and the type is what says whether a bead is polar.  The
table is the most common type martinize gives each bead over the benchmark
proteins; backbone types change with secondary structure, but not between
polar and apolar classes for the residues where the majority is clear.

    python make_bead_types.py --data ~/Dropbox/PocketFinding/SchrodingerSet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cgprep  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    import boonza

    files = sorted((args.data / "apo").glob("*.mae"))
    table = {}
    for model in cgprep.FORCEFIELDS:
        counts = defaultdict(Counter)
        for f in files:
            cg = cgprep.coarse_grain(boonza.load(str(f)), model, "protein and not resname LIG")
            res = np.asarray(cg.atoms["residue"])
            resnames = np.asarray(cg.residues["name"])
            for a in range(cg.natoms):
                key = (cgprep.standard_resname(resnames[res[a]]), str(cg.atoms["name"][a]))
                counts[key][str(cg.atoms["type"][a])] += 1
        out = defaultdict(dict)
        for (resname, bead), c in sorted(counts.items()):
            out[resname][bead] = c.most_common(1)[0][0]
            if any(len({cgprep.martini_polar(t, n) for t in c}) > 1 for n in (False, True)):
                print(f"{model} {resname} {bead}: polarity depends on structure {dict(c)}")
        table[model] = out
        print(f"{model}: {sum(len(v) for v in out.values())} beads from {len(files)} proteins")
    (Path(__file__).resolve().parents[1] / "bead_types.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
