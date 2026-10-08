"""Build a benchmark set: fpocket-ready PDBs for every representation, and its truth.

Three sets are known:

``pp48``
    The 48 apo/holo pairs fpocket was validated on (fpocket-data-1.0, from
    the LIGSITE-csc benchmark); the apo structures come superposed on their
    holo partners.  They are raw crystal structures, so ``--fix`` rebuilds
    missing side-chain heavy atoms with PDBFixer first (no missing loops, no
    hydrogens, numbering kept), and every representation is
    made from the rebuilt structure.  The holo site is every protein residue within
    ``SITE_CUTOFF`` of the ligand; the apo site is the apo residues the
    sequence alignment (boonza's matchmaker) pairs with them.
``train263``
    fpocket's own training set (fpocket-data-1.0, data/train): 263 holo
    complexes, on which fpocket's flags and pocket score were fitted.  The
    site is every protein residue within ``SITE_CUTOFF`` of the ligand.
``schrodinger``
    The apo/holo pairs of ``si.csv`` not marked EXCLUDE.  The holo ligand
    (resname LIG) is carried onto the apo by boonza.superpose of the holo
    protein on the apo protein (alpha carbons paired by sequence, pruned as
    ChimeraX does).  si.csv's residue columns are not used.

Only protein is kept: waters, ions, cofactors and the ligand are removed.
``data/<set>/cases.json`` holds one entry per scored structure (its site
residues and ligand atoms), ``data/<set>/pdb`` the structures.

    conda activate boonza
    PYTHONPATH=/path/to/pdbfixer python prepare.py pp48 --data data/pp48 --fix
    python prepare.py schrodinger --data ~/Dropbox/PocketFinding/SchrodingerSet
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from boonza.pockets import beads, prepare as cgprepare

HERE = Path(__file__).resolve().parent
SITE_CUTOFF = 5.0  # Å: residues this close to a ligand heavy atom make the site


def labels_of(system, residues):
    chains = np.asarray(system.chains["name"])
    return [(str(chains[system.residues["chain"][r]]), int(system.residues["resid"][r]),
             str(system.residues["insertion"][r]).strip()) for r in residues]  # fmt: skip


def ligand_names(resname: str) -> list[str]:
    return [n.upper() for n in re.split(r"[ ,]+", resname) if n]


def ligand_atoms(system, resname: str) -> np.ndarray:
    res = np.asarray(system.atoms["residue"])
    rn = np.array([str(x).strip().upper() for x in system.residues["name"]])
    heavy = np.asarray(system.atoms["anum"]) > 1
    ids = np.flatnonzero(np.isin(rn[res], ligand_names(resname)) & heavy)
    if not len(ids):
        raise ValueError(f"no ligand {resname!r}")
    return ids


def protein_selection(resname: str) -> str:
    """Protein, without the ligand: amino-acid ligands read as protein."""
    names = " ".join(f'"{n}"' for n in ligand_names(resname))  # quoted: a code can be numeric
    return f"protein and not resname {names}"


def residues_near(system, selection: str, xyz: np.ndarray, cutoff: float) -> list[int]:
    ids = system.select(f"({selection}) and not hydrogen").ids
    d = np.linalg.norm(np.asarray(system.positions)[ids][:, None] - xyz[None], axis=2).min(1)
    return sorted(set(np.asarray(system.atoms["residue"])[ids[d < cutoff]].tolist()))


def aligned_residues(apo, holo, holo_residues: list[int]) -> list[int]:
    """The apo residues sequence-aligned with ``holo_residues`` (chain by chain)."""
    import boonza

    atom_res = np.asarray(holo.atoms["residue"])
    wanted = set(holo_residues)
    out = set()
    for hc in sorted({int(holo.residues["chain"][r]) for r in holo_residues}):
        try:
            m = boonza.matchmaker(apo, holo, reference_chain=hc, apply=False, cutoff=None)
        except ValueError:
            continue
        for a, h in zip(m.mobile_atoms, m.reference_atoms, strict=True):
            if int(atom_res[h]) in wanted:
                out.add(int(apo.atoms["residue"][a]))
    return sorted(out)


# ---------------------------------------------------------------- the sets


def fix_pdb(args) -> str:
    """Rebuild missing heavy atoms of ``src`` into ``dst`` (PDBFixer).

    PDBFixer drops some groups (an adenine named ADE, a sucrose written as
    ATOM records), and the ligand is the truth: every residue PDBFixer did
    not write back is copied from ``src`` unchanged.
    """
    src, dst = args
    if Path(dst).exists():
        return dst
    import io

    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=src)
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # leave gaps as they are: no invented loops
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    text = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, text, keepIds=True)
    def residue(line):  # chain, number, insertion code, name
        return line[21], line[22:26], line[26], line[17:20].strip()

    fixed = [x for x in text.getvalue().splitlines() if x.startswith(("ATOM", "HETATM", "TER"))]
    kept = {residue(x) for x in fixed if not x.startswith("TER")}
    dropped = [x.rstrip("\n") for x in open(src)
               if x.startswith(("ATOM", "HETATM")) and residue(x) not in kept]  # fmt: skip
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("\n".join(fixed + dropped + ["END"]) + "\n")
    return dst


def tpocket_pairs(data: Path, listing: str, prefix: str, source: str = "raw") -> list[dict]:
    """The structures of a tpocket input file (apo, holo, ligand per line), with the
    directory it names (``prefix``) found as ``data/<source>``.  A line whose apo
    and holo are the same file is a holo structure alone."""
    pairs = []
    for line in open(data / listing):
        if not line.strip():
            continue
        apo, holo, lig = line.split()[:3]
        apo = data / apo.replace(prefix, f"{source}/")
        holo = data / holo.replace(prefix, f"{source}/")
        p = {"case": holo.stem.upper(), "holo_file": str(holo), "holo": holo.stem.upper(), "ligand": lig}
        if apo != holo:
            p.update(apo_file=str(apo), apo=apo.stem.upper())
        pairs.append(p)
    return pairs


def pp48_pairs(data: Path, source: str = "raw") -> list[dict]:
    """``data`` holds pp-apo-t.txt and, as raw/, the pp_data directory it names."""
    return tpocket_pairs(data, "pp-apo-t.txt", "data/pp_data/", source)


def train_pairs(data: Path, source: str = "raw") -> list[dict]:
    """fpocket's training set: ``data`` holds train-t.txt and, as raw/, data/train."""
    return tpocket_pairs(data, "train-t.txt", "data/train/", source)


def holo_truth(p: dict) -> list[dict]:
    """A holo structure alone: its site is the residues near its ligand."""
    import boonza

    holo = boonza.load(p["holo_file"])
    lig = np.asarray(holo.positions)[ligand_atoms(holo, p["ligand"])]
    site = residues_near(holo, protein_selection(p["ligand"]), lig, SITE_CUTOFF)
    return [{"case": p["case"], "kind": "holo", "stem": f"holo_{p['holo']}",
             "site": labels_of(holo, site), "ligand": lig.round(3).tolist(),
             "note": f"{len(site)} site residues"}]  # fmt: skip


def pp48_truth(p: dict) -> list[dict]:
    import boonza

    apo, holo = boonza.load(p["apo_file"]), boonza.load(p["holo_file"])
    protein = protein_selection(p["ligand"])
    lig = np.asarray(holo.positions)[ligand_atoms(holo, p["ligand"])]
    holo_site = residues_near(holo, protein, lig, SITE_CUTOFF)
    apo_site = aligned_residues(apo, holo, holo_site)
    if not apo_site:  # no alignment: the residues near the (superposed) ligand
        apo_site = residues_near(apo, protein, lig, SITE_CUTOFF)
    note = f"{len(holo_site)} holo / {len(apo_site)} apo site residues"
    return [
        {"case": p["case"], "kind": "apo", "stem": f"apo_{p['apo']}",
         "site": labels_of(apo, apo_site), "ligand": lig.round(3).tolist(), "note": note},
        {"case": p["case"], "kind": "holo", "stem": f"holo_{p['holo']}",
         "site": labels_of(holo, holo_site), "ligand": lig.round(3).tolist(), "note": note},
    ]  # fmt: skip


def schrodinger_pairs(data: Path) -> list[dict]:
    rows = list(csv.reader(open(data / "si.csv", encoding="utf-8-sig")))
    pairs = []
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        apo, holo, _, note = (x.strip() for x in row[:4])
        apo_file = data / "apo" / f"{apo.lower()}.mae"
        holo_file = data / "holo" / f"{holo.lower()}.mae"
        if "EXCLUDE" in note.upper() or not apo_file.exists() or not holo_file.exists():
            continue
        pairs.append({"case": f"{apo}_{holo}", "apo_file": str(apo_file), "holo_file": str(holo_file),
                      "apo": apo.lower(), "holo": holo.lower(), "ligand": "LIG",
                      "note": note})  # fmt: skip
    return pairs


def schrodinger_truth(p: dict) -> list[dict]:
    """The holo ligand on each structure: as it is on the holo, and on the apo
    carried by boonza.superpose of the whole holo protein on the apo protein
    (results.ligand_on).  The residue lists of si.csv are not used."""
    import boonza

    import results

    apo, holo = boonza.load(p["apo_file"]), boonza.load(p["holo_file"])
    apo_protein = apo.select(f"protein and {cgprepare.NOT_PROBES}").clone()
    lig_holo = np.asarray(holo.positions)[ligand_atoms(holo, "LIG")]
    lig_apo, sup = results.ligand_on(holo, apo_protein, with_fit=True)
    fit = (f"holo superposed on apo ({sup.n_used}/{sup.n_matched} alpha carbons, "
           f"RMSD {sup.rmsd:.2f} Å)")  # fmt: skip
    note = f"{p['note']}; {fit}".lstrip("; ")
    # the residues within SITE_CUTOFF of the ligand, for reference only (no criterion uses them)
    apo_site = labels_of(apo, residues_near(apo, protein_selection("LIG"), lig_apo, SITE_CUTOFF))
    holo_site = labels_of(holo, residues_near(holo, protein_selection("LIG"), lig_holo, SITE_CUTOFF))
    return [
        {"case": p["case"], "kind": "apo", "stem": f"apo_{p['apo']}", "site": apo_site,
         "ligand": lig_apo.round(3).tolist(), "note": note},
        {"case": p["case"], "kind": "holo", "stem": f"holo_{p['holo']}", "site": holo_site,
         "ligand": lig_holo.round(3).tolist(), "note": note},
    ]  # fmt: skip


SETS = {"pp48": (pp48_pairs, pp48_truth), "schrodinger": (schrodinger_pairs, schrodinger_truth),
        "train263": (train_pairs, holo_truth)}

# ---------------------------------------------------------------- structures


def write_structure(args):
    path, out_stem, protein = args
    import boonza

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = boonza.load(str(path))
        done = {}
        for model in beads.MODELS:
            out = Path(f"{out_stem}_{model}.pdb")
            npolar = Path(f"{out_stem}_{model}_npolar.pdb")  # Martini N beads counted polar
            try:
                if not out.exists() or (model in beads.FORCEFIELDS and not npolar.exists()):
                    cg = cgprepare.coarse_grain(s, model, protein)
                    beads.write_fpocket_pdb(cg, out, model=model)
                    if model in beads.FORCEFIELDS:
                        beads.write_fpocket_pdb(cg, npolar, model=model, n_polar=True)
                done[model] = True
            except Exception as e:  # noqa: BLE001
                done[model] = f"{type(e).__name__}: {e}"
    notes = sorted({str(w.message) for w in caught if "left out" in str(w.message)})
    return str(path), done, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--data", type=Path, required=True, help="where the set's files are")
    ap.add_argument("--out", type=Path, help="default: data/<set>")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--fix", action="store_true", help="pp48: rebuild missing heavy atoms first")
    args = ap.parse_args()

    out = args.out or HERE / "data" / args.set
    (out / "pdb").mkdir(parents=True, exist_ok=True)
    read_pairs, truth = SETS[args.set]
    if args.fix:
        if args.set not in ("pp48", "train263"):
            sys.exit("--fix is for the raw crystal structures of pp48 and train263")
        raw = read_pairs(args.data)
        todo = sorted({(p[f"{k}_file"], p[f"{k}_file"].replace("/raw/", "/fixed/"))
                       for p in raw for k in ("apo", "holo") if f"{k}_file" in p})  # fmt: skip
        with ProcessPoolExecutor(args.jobs) as pool:
            list(pool.map(fix_pdb, todo))
        pairs = read_pairs(args.data, "fixed")
    else:
        pairs = read_pairs(args.data)

    with ProcessPoolExecutor(args.jobs) as pool:
        entries = [e for es in pool.map(truth, pairs) for e in es]
        for e in entries:
            print(f"{e['kind']:4s} {e['case']:12s} {len(e['site']):3d} site residues, "
                  f"{len(e['ligand']):3d} ligand atoms  {e['note']}")  # fmt: skip
        jobs = {(p[f"{k}_file"], str(out / "pdb" / f"{k}_{p[k]}"), protein_selection(p["ligand"]))
                for p in pairs for k in ("apo", "holo") if f"{k}_file" in p}  # fmt: skip
        failed = {}
        for path, done, notes in pool.map(write_structure, sorted(jobs)):
            bad = {m: v for m, v in done.items() if v is not True}
            if bad:
                failed[path] = bad
                print(f"FAILED {path}: {bad}")
            for n in notes:
                print(f"{Path(path).name}: {n}")
    (out / "cases.json").write_text(json.dumps({"structures": entries, "failed": failed}, indent=1))
    print(f"{len(pairs)} pairs, {len(jobs)} structures, {len(failed)} with failures")


if __name__ == "__main__":
    main()
