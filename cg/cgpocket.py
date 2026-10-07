"""fpocket and mdpocket on coarse-grained proteins (Martini 2, Martini 3, SIRAH).

    conda activate boonza

    # an all-atom structure, mapped onto beads and searched with the CG preset
    python cgpocket.py run protein.mae --model martini3 -o out/

    # a structure that is coarse-grained already (types come from the topology)
    python cgpocket.py run cg.gro --top topol.top --model martini3 -o out/

    # a CG trajectory: protein beads only, made whole, fitted, then mdpocket
    python cgpocket.py traj --top topol.top --coords cg.gro --traj md.xtc \\
        --model martini3 -o out/md --run

    # check the pockets against a known ligand: where it sits in a holo structure
    python cgpocket.py run apo.mae --model sirah -o out/ \\
        --holo holo.mae --holo-ligand "resname LIG"

    # the tuned flags, to pass to fpocket or mdpocket yourself
    python cgpocket.py flags --model sirah

The flags and pocket score were tuned as fpocket's own were, on its
263-complex training set, and tested on fpocket's 48 apo/holo pairs and the
Schrodinger apo set (benchmark/); see ``presets.json`` and README.md.  The
presets pass --score_coefficients, which needs the fpocket build in ../bin.
Extra fpocket flags after ``--`` override the preset.

mdpocket contours its density map at 8, a level set for all-atom proteins;
``traj --run`` also writes the contour at the model's calibrated level
(``<prefix>_dens_iso_<level>.pdb``).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cgprep  # noqa: E402
import core  # noqa: E402
import volume as voxels  # noqa: E402

PRESETS = HERE / "presets.json"

#: fpocket's own hit criteria, as tpocket applies them
PPC_CUTOFF = 4.0  # Å: PPc, pocket center to the nearest ligand atom
MOC_D = 3.0  # Å: MOc contact distance between ligand atoms and sphere centers
MOC_LIGAND = 0.5  # MOc: more than this share of ligand atoms near the pocket's spheres
MOC_POCKET = 0.2  # MOc: more than this share of the pocket's spheres near the ligand
BEAD_TYPES = HERE / "bead_types.json"

#: residue names of protein beads, for CG systems the "protein" keyword misses
AMINO_ACIDS = set(cgprep._SIRAH_RESNAMES) | set(cgprep._STANDARD_RESNAMES) | {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}  # fmt: skip


def binary(name: str) -> str:
    local = HERE.parent / "bin" / name
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if not found:
        sys.exit(f"cannot find {name}: build fpocket (make) or put it on PATH")
    return found


def preset(model: str) -> list[str]:
    presets = json.loads(PRESETS.read_text())
    if model not in presets:
        sys.exit(f"no preset for {model!r}; presets: {', '.join(presets)}")
    if presets[model].get("flags") is None:
        sys.exit(f"the {model} preset is not ready: {presets[model].get('pending', 'no flags')}")
    return list(presets[model]["flags"])


def n_polar(model: str) -> bool:
    presets = json.loads(PRESETS.read_text())
    return bool(presets.get(model, {}).get("n_polar", False))


def merge_flags(base: list[str], extra: list[str]) -> list[str]:
    """``base`` with any flag ``extra`` repeats replaced by ``extra``'s value."""
    extra = [x for x in extra if x != "--"]
    given = {extra[k] for k in range(len(extra)) if extra[k].startswith("-")}
    out, k = [], 0
    while k < len(base):
        if base[k] in given:
            k += 2
            continue
        out.append(base[k])
        k += 1
    return out + extra


def load(path, top=None):
    import boonza  # noqa: PLC0415

    if top is not None:
        from boonza.io import load_top  # noqa: PLC0415

        return load_top(top, coordinates=path, structure_only=True)
    return boonza.load(str(path))


def protein_ids(system, selection: str | None = None) -> np.ndarray:
    """The protein atoms or beads, or those of ``selection``, never the probes.

    Probes -- ligand copies in chain LIG -- are often amino acids themselves,
    which the protein keyword and the residue names would both take."""
    if selection:
        ids = system.select(selection).ids
    else:
        ids = system.select("protein").ids
        if not len(ids):
            res = np.asarray(system.atoms["residue"])
            names = np.asarray(system.residues["name"])
            ids = np.flatnonzero(np.isin(names[res], list(AMINO_ACIDS)))
    probes = cgprep.probe_ids(system)
    if len(np.intersect1d(ids, probes)):
        print(f"leaving out {len(probes)} probe atoms/beads (chain {cgprep.PROBE_CHAIN})",
              file=sys.stderr)  # fmt: skip
    return np.setdiff1d(ids, probes)


def kind(model: str) -> str:
    """How bead polarity is read: from elements, SIRAH names, or Martini types."""
    return model if model in ("aa", "sirah") else "martini"


def fill_types(system, model: str) -> None:
    """Martini bead types from the bundled table, for a system read without its topology."""
    types = [str(t) for t in system.atoms["type"]]
    if all(types):
        return
    table = json.loads(BEAD_TYPES.read_text())[model]
    res = np.asarray(system.atoms["residue"])
    names = np.asarray(system.atoms["name"])
    resnames = np.asarray(system.residues["name"])
    missing = set()
    for a in range(system.natoms):
        if not types[a]:
            r = cgprep.standard_resname(resnames[res[a]])
            t = table.get(r, {}).get(str(names[a]))
            if t is None:
                missing.add(f"{r}:{names[a]}")
                t = "P1"
            system.atoms["type"][a] = t
    if missing:
        print(f"warning: no bead type known for {', '.join(sorted(missing))}; taken as polar",
              file=sys.stderr)  # fmt: skip


def fpocket_ready(system, model: str, selection: str | None = None):
    """(the system to write, the ids of its protein atoms or beads)."""
    ids = protein_ids(system, selection)
    if not len(ids):
        sys.exit("no protein atoms or beads selected")
    names = np.asarray(system.atoms["name"])[ids]
    resnames = np.asarray(system.residues["name"])[np.asarray(system.atoms["residue"])[ids]]
    found = cgprep.guess_model(names, resnames)
    if found == "aa" and model != "aa":  # all-atom input: map it onto beads
        atoms = f"({selection or 'protein'}) and {cgprep.NOT_PROBES}"
        system = cgprep.coarse_grain(system, model, atoms)
        return system, np.arange(system.natoms)
    if found == "sirah" and model != "sirah" or found == "martini" and not model.startswith("martini"):
        sys.exit(f"the input looks like {found}, not {model}")
    if model.startswith("martini"):
        fill_types(system, model)
    return system, ids


def cmd_prep(args, extra=None):
    system, ids = fpocket_ready(load(args.input, args.top), args.model, args.selection)
    out = Path(args.output)
    if out.suffix != ".pdb":
        out.mkdir(parents=True, exist_ok=True)
        out = out / f"{Path(args.input).stem}_{args.model}.pdb"
    cgprep.write_fpocket_pdb(system, out, ids, kind(args.model), n_polar(args.model))
    print(f"wrote {out} ({len(ids)} {'atoms' if args.model == 'aa' else 'beads'})")
    return out


def cmd_run(args, extra):
    pdb = cmd_prep(args)
    flags = merge_flags(preset(args.model), extra)
    cmd = [binary("fpocket"), "-f", str(pdb), *flags]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"pockets in {pdb.with_name(pdb.stem + '_out')}")
    apo_path, same_frame = args.apo, False
    if apo_path is None and cgprep.guess_model(*_names_resnames(load(args.input, args.top))) == "aa":
        apo_path, same_frame = args.input, True  # the beads were mapped from it
    apo = all_atom_apo(apo_path, pdb, same_frame) if apo_path else None
    ligand_sel = holo = rows = None
    if args.holo:
        ligand_sel, holo, rows = compare_to_holo(pdb, args.holo, args.holo_ligand, args.holo_top, apo)
    if apo is not None:
        view = write_view(pdb, apo, holo, ligand_sel, rows, args.view_dir,
                          structure_format(apo_path), structure_format(args.holo or apo_path))
        print(f"\nall-atom view: cd {view.parent} && pymol {view.name}")
    else:
        print("\n(no all-atom view: give the all-atom apo structure with --apo)")


def _names_resnames(system):
    ids = protein_ids(system)
    names = np.asarray(system.atoms["name"])[ids]
    resnames = np.asarray(system.residues["name"])[np.asarray(system.atoms["residue"])[ids]]
    return names, resnames


def structure_format(path) -> str:
    """The format a structure was given in, to write it back in the same one (MAE stays
    MAE, bond orders and all; CIF stays CIF, ...).  DMS is written as MAE, which keeps
    the same structure and bonds and which PyMOL opens; PDB only for formats boonza
    cannot write."""
    from boonza.io import _WRITERS  # noqa: PLC0415

    suffixes = [x.lower() for x in Path(str(path)).suffixes]
    if suffixes and suffixes[-1] in (".gz", ".bz2"):  # compressed: the format underneath
        suffixes = suffixes[:-1]
    ext = {".maegz": ".mae", ".cmsgz": ".cms", ".dms": ".mae"}.get(suffixes[-1] if suffixes else "",
                                                                   suffixes[-1] if suffixes else "")
    return ext[1:] if ext in _WRITERS else "pdb"


def read_pockets(out_dir: Path, stem: str):
    """[(alpha-sphere centers (n, 3), score)] in fpocket's ranking."""
    spheres: dict[int, list] = {}
    for line in open(out_dir / f"{stem}_pockets.pqr"):
        if line.startswith("ATOM"):
            spheres.setdefault(int(line[22:26]), []).append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    scores = [float(x.split(":")[1]) for x in open(out_dir / f"{stem}_info.txt")
              if x.strip().startswith("Score :")]  # fmt: skip
    return [(np.array(v), scores[k - 1] if k <= len(scores) else float("nan"))
            for k, v in sorted(spheres.items())]  # fmt: skip


def backbone_name(system) -> str:
    """What the structure calls its backbone: CA, Martini's BB or SIRAH's GC."""
    names = set(np.asarray(system.atoms["name"]).tolist())
    found = next((n for n in ("CA", "BB", "GC") if n in names), None)
    if found is None:
        sys.exit("no CA, BB or GC atoms to superpose on")
    return found


def without_probes(system, ligand: str | None = None):
    """``system`` without its chain-LIG probes -- unless ``ligand`` lies there."""
    probes = cgprep.probe_ids(system)
    if not len(probes):
        return system
    lig_ids = system.select(f"({ligand})").ids if ligand else np.array([], int)
    if len(np.intersect1d(lig_ids, probes)):
        print(f"warning: the ligand {ligand!r} is in chain {cgprep.PROBE_CHAIN}, "
              "which holds probes by convention", file=sys.stderr)  # fmt: skip
        return system
    return system.select(cgprep.NOT_PROBES).clone()


def all_atom_apo(apo_path, pdb: Path, same_frame: bool):
    """The all-atom apo protein in the frame of the pockets (the structure in ``pdb``).

    ``same_frame``: ``pdb`` was mapped from this very structure, so its frame
    is already the pockets'.  Otherwise its alpha carbons are superposed on
    the backbone beads (or atoms) of ``pdb`` by sequence."""
    import boonza  # noqa: PLC0415

    apo = load(apo_path)
    apo = apo.select(f"protein and {cgprep.NOT_PROBES}").clone()
    if not same_frame:
        ref = load(pdb)
        sup = boonza.superpose(apo, ref, sel="name CA", ref_sel=f"name {backbone_name(ref)}",
                               match="sequence", apply=True)  # fmt: skip
        print(f"all-atom apo superposed on the pockets' structure: {sup.n_used}/{sup.n_matched} "
              f"alpha carbons, RMSD {sup.rmsd:.2f} Å")  # fmt: skip
    return apo


def place_holo(holo, reference, ligand: str | None):
    """(ligand heavy atoms, fit info, the whole holo structure) in ``reference``'s frame.

    boonza.superpose fits the holo protein's alpha carbons on the reference's
    backbone (CA, or Martini BB / SIRAH GC beads), paired by sequence and
    pruned of pairs that stay more than 2 Å apart (as ChimeraX matchmaker
    does); the whole holo structure, ligand included, moves with the fit.
    """
    import boonza  # noqa: PLC0415

    ligand = ligand or "resname LIG"
    moved = holo.clone()
    try:
        sup = boonza.superpose(moved, reference, sel=f"protein and name CA and {cgprep.NOT_PROBES}",
                               ref_sel=f"name {backbone_name(reference)} and {cgprep.NOT_PROBES}",
                               match="sequence", apply=True)  # fmt: skip
    except ValueError as e:
        sys.exit(f"cannot superpose the holo structure: {e}")
    lig = np.asarray(moved.positions)[moved.select(f"({ligand}) and not element H").ids]
    if not len(lig):
        sys.exit(f"the holo structure has no {ligand}")
    info = {"ligand": ligand, "atoms": len(lig), "paired": sup.n_used,
            "matched": sup.n_matched, "fit_rmsd": sup.rmsd}  # fmt: skip
    return lig, info, moved


def compare_to_holo(pdb: Path, holo_path, ligand: str | None, holo_top=None, apo=None):
    """Score every pocket fpocket found in ``pdb`` against the ligand of a holo structure.

    The holo protein is superposed (boonza.superpose) on the all-atom ``apo``
    when there is one (alpha carbon on alpha carbon), else on ``pdb`` itself
    (beads or atoms), and the ligand carried into that frame -- the pockets'
    frame either way.
    For each pocket: the distance from its center (the mean of its
    alpha-sphere centers) to the nearest ligand atom and to the ligand's
    centroid, and the two shares MOc tests; PPc and MOc as tpocket applies
    them.  Returns (ligand selection, the holo structure moved, the rows).
    """
    import importlib  # noqa: PLC0415

    sites = importlib.import_module("boonza.sites")
    holo = without_probes(load(holo_path, holo_top), ligand)
    reference = apo if apo is not None else load(pdb)
    lig, info, moved = place_holo(holo, reference, ligand)
    onto = "the all-atom apo" if apo is not None else pdb.name
    print(f"\nholo ligand {info['ligand']}: {info['atoms']} heavy atoms; holo superposed on {onto} "
          f"({info['paired']}/{info['matched']} alpha carbons kept, RMSD {info['fit_rmsd']:.2f} Å)")
    if info["paired"] < 20 or info["fit_rmsd"] > 3.0:
        print("warning: a poor superposition; distances below are unreliable", file=sys.stderr)

    out_dir = pdb.with_name(pdb.stem + "_out")
    rows = []
    for rank, (centers, score) in enumerate(read_pockets(out_dir, pdb.stem), 1):
        d = np.linalg.norm(lig[:, None] - centers[None], axis=2) < MOC_D
        lig_cov, pocket_cov = float(d.any(1).mean()), float(d.any(0).mean())
        dca, dcc = sites.dca(centers, lig), sites.dcc(centers, lig)
        rows.append((rank, score, dca, dcc, lig_cov, pocket_cov, dca < PPC_CUTOFF,
                     lig_cov > MOC_LIGAND and pocket_cov > MOC_POCKET))  # fmt: skip
    table = out_dir / f"{pdb.stem}_vs_holo.csv"
    with open(table, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["rank", "score", "center_to_nearest_ligand_atom", "center_to_ligand_centroid",
                      "ligand_atoms_within_3A", "spheres_within_3A", "PPc", "MOc"])  # fmt: skip
        for r in rows:
            out.writerow([f"{x:.3f}" if isinstance(x, float) else str(x) for x in r])
    first = {name: next((r[0] for r in rows if r[col]), None) for name, col in (("PPc", 6), ("MOc", 7))}
    shown = {k: (f"rank {v}" if v else "none") for k, v in first.items()}
    print(f"first correct pocket: PPc {shown['PPc']}, MOc {shown['MOc']} (of {len(rows)} pockets)")
    print("rank  score   center->ligand atom  center->ligand centroid  ligand covered  spheres on ligand  PPc  MOc")
    for r in rows[:10]:
        print(f"{r[0]:4d} {r[1]:7.3f} {r[2]:15.1f} Å {r[3]:19.1f} Å {r[4]:14.0%} {r[5]:17.0%}"
              f"  {'yes' if r[6] else ' - '}  {'yes' if r[7] else ' - '}")  # fmt: skip
    print(f"every pocket: {table}")
    return info["ligand"], moved, rows


#: pocket colours by rank in the view; the others are grey
RANK_COLORS = ("red", "orange", "yellow", "green", "cyan")


def pocket_object_lines(source: str, pockets) -> list[str]:
    """PyMOL lines making each pocket one object, ``pocket_<number>``, to show or hide alone.

    ``source``: the object the pockets' alpha spheres were loaded into, one residue
    number per pocket.  ``pockets``: [(number, label, centre)].  Each pocket object
    holds its alpha spheres (scaled by 0.3, as fpocket's own script draws them,
    coloured by rank: 1 red, 2 orange, 3 yellow, 4 green, 5 cyan, the rest grey) and,
    as one more atom named LBL at its centre, its label -- so one click hides or shows
    both.  The pockets not drawn go with ``source``."""
    lines, names = [], []
    for k, label, (x, y, z) in pockets:
        color = RANK_COLORS[k - 1] if 1 <= k <= len(RANK_COLORS) else "grey50"
        obj = f"pocket_{k}"
        lines += [f"create {obj}, {source} and resi {k}",
                  f'pseudoatom {obj}, name=LBL, resi={k}, pos=[{x:.3f}, {y:.3f}, {z:.3f}], label="{label}"',
                  f"hide everything, {obj}",  # PyMOL bonds nearby points of a .pqr
                  f"set sphere_scale, 0.3, {obj}", f"set sphere_transparency, 0.2, {obj}",
                  f"show spheres, {obj} and not name LBL", f"color {color}, {obj}",
                  f"show labels, {obj} and name LBL"]  # fmt: skip
        names.append(obj)
    lines += [f"delete {source}",
              "set label_size, 18", "set label_color, white", "set label_position, (0, 0, 8)"]  # fmt: skip
    return lines


def write_view(pdb: Path, apo, holo=None, ligand: str | None = None, rows=None,
               view_dir=None, fmt: str = "pdb", holo_fmt: str | None = None) -> Path:  # fmt: skip
    """A PyMOL script showing the pockets on the all-atom structures, beads hidden.

    The apo protein as a translucent cartoon, the holo ligand as blue sticks
    (the aligned holo protein, light blue, loaded but hidden, to toggle on), and every pocket's alpha-sphere centers coloured by rank
    (1 red, 2 orange, 3 yellow, 4 green, 5 cyan, the rest grey), each labelled
    with its rank -- and with PPc/MOc when a holo ligand was given.

    ``view_dir``: write the script and its files there under plain names
    (view.pml, apo.<fmt>, holo.<fmt>, ligand.<fmt>, pockets.pqr, ...) rather
    than beside fpocket's output.  Each structure is written in the format it
    was read in (``fmt`` for the apo, ``holo_fmt`` for the holo and its
    ligand): MAE stays MAE, keeping bond orders."""
    import shutil as sh  # noqa: PLC0415

    from boonza.io import save as save_structure  # noqa: PLC0415

    out_dir = pdb.with_name(pdb.stem + "_out")
    stem = pdb.stem
    if view_dir is not None:
        where = Path(view_dir)
        where.mkdir(parents=True, exist_ok=True)
        ext = {"apo": fmt, "holo": holo_fmt or fmt, "ligand": holo_fmt or fmt}
        name = {k: f"{k}.{e}" for k, e in ext.items()}
        name.update(pockets="pockets.pqr", view="view.pml")
        sh.copy(out_dir / f"{stem}_pockets.pqr", where / name["pockets"])
        for src, dst in ((f"{stem}_info.txt", "fpocket_info.txt"), (f"{stem}_vs_holo.csv", "pockets_vs_holo.csv")):
            if (out_dir / src).exists():
                sh.copy(out_dir / src, where / dst)
    else:
        where = out_dir
        ext = {"apo": fmt, "holo": holo_fmt or fmt, "ligand": holo_fmt or fmt}
        name = {k: f"{stem}_{k}.{e}" for k, e in ext.items()}
        name.update(pockets=f"{stem}_pockets.pqr", view=f"{stem}_view.pml")
    save_structure(apo, where / name["apo"])
    lines = [f"# pockets fpocket found on {stem}, shown on the all-atom structures (cgpocket)",
             f"load {name['apo']}, apo"]  # fmt: skip
    if holo is not None:
        save_structure(holo.select(f"not ({ligand})").clone(), where / name["holo"])
        save_structure(holo.select(f"({ligand})").clone(), where / name["ligand"])
        lines += [f"load {name['holo']}, holo", f"load {name['ligand']}, ligand"]
    lines += [f"load {name['pockets']}, pockets_all",
              "hide everything",
              "show cartoon, apo", "color wheat, apo", "set cartoon_transparency, 0.45, apo"]  # fmt: skip
    if holo is not None:
        lines += ["show cartoon, holo", "color lightblue, holo",  # the ligand's family (blue), apart from the apo's wheat
                  "# click 'holo' in the object panel to compare the folds",
                  "disable holo",
                  "show sticks, ligand and not hydro", "color tv_blue, ligand and elem C",
                  "set stick_radius, 0.3, ligand"]  # fmt: skip
    pockets = read_pockets(out_dir, stem)
    verdict = {r[0]: r for r in rows or []}
    drawn = []
    for k, (centers, _) in enumerate(pockets, 1):
        tag = str(k)
        if k in verdict:
            tag += " PPc" if verdict[k][6] else ""
            tag += " MOc" if verdict[k][7] else ""
        drawn.append((k, tag, centers.mean(0)))
    lines += ["# each pocket is the object pocket_<rank>, its alpha spheres and its label: click it",
              "# in the object panel to show or hide it"]
    lines += pocket_object_lines("pockets_all", drawn)
    lines += ["bg_color black", "set ray_opaque_background, 0"]  # exported images: transparent
    focus = next((r[0] for r in rows or [] if r[6]), 1)
    lines += [f"zoom pocket_{focus}, 12" if pockets else "orient apo"]
    view = where / name["view"]
    view.write_text("\n".join(x for x in lines if x) + "\n")
    return view


def cmd_traj(args, extra):
    import boonza  # noqa: PLC0415

    system = load(args.coords, args.top) if args.top else load(args.coords)
    model = args.model
    if model.startswith("martini"):
        fill_types(system, model)
    ids = protein_ids(system, args.selection)
    if cgprep.guess_model(*_names_resnames(system)) == "aa" and model != "aa":
        sys.exit("traj needs a coarse-grained topology: the trajectory's own beads")
    traj = boonza.open_trajectory(str(args.traj), system)
    view = traj[args.start : args.stop : args.stride]
    glue = None
    if not args.no_fix:
        if args.fit:  # never the probes, whatever the selection says
            fit = np.setdiff1d(system.select(args.fit).ids, cgprep.probe_ids(system))
        else:  # the protein's backbone beads (BB, GC) or alpha carbons
            names = np.asarray(system.atoms["name"])
            fit = ids[names[ids] == backbone_name(system.select(f"index {' '.join(map(str, ids))}").clone())]
        print(f"fitting every frame on the first by {len(fit)} beads/atoms", flush=True)
        whole = ids if system.nbonds else None
        glue = boonza.Glue(system, glue=[ids], center=ids, fit=fit, whole=whole)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    first = None
    with boonza.open_writer(str(prefix) + ".dcd", len(ids)) as w:
        blocks = glue.frames(view) if glue else view.chunks()
        n = 0
        for block in blocks:
            for k in range(len(block)):
                pos = block.positions[k][ids]
                if first is None:
                    first = pos
                w.write(pos, None)
                n += 1
    s = system.clone() if hasattr(system, "clone") else system
    s.positions[ids] = first
    cgprep.write_fpocket_pdb(s, str(prefix) + ".pdb", ids, kind(model), n_polar(model))
    print(f"wrote {prefix}.pdb and {prefix}.dcd: {len(ids)} beads, {n} frames")
    flags = merge_flags(preset(model), extra)
    cmd = [binary("mdpocket"), "-f", f"{prefix}.pdb", "-t", f"{prefix}.dcd", "-O", "dcd",
           "-o", str(prefix), *flags]  # fmt: skip
    print(" ".join(cmd), flush=True)
    if args.run:
        subprocess.run(cmd, check=True)
        iso = json.loads(PRESETS.read_text())[model].get("mdpocket_density_iso")
        if iso is not None:
            out = write_isosurface(Path(f"{prefix}_dens.dx"), iso)
            print(f"\ndensity contoured at {iso} for {model} (mdpocket's 8 is for all-atom): {out}")
        pdb = Path(f"{prefix}.pdb")  # the reference frame of the maps
        where = Path(args.view_dir) if args.view_dir else prefix.parent
        apo = all_atom_apo(args.apo, pdb, same_frame=False) if args.apo else None
        reference = apo if apo is not None else load(pdb)
        holos, ligands = [], []
        for path in args.holo or []:
            tag = Path(str(path)).name.split(".")[0]
            holo_sys = without_probes(load(path, args.holo_top), args.holo_ligand)
            lig, info, moved = place_holo(holo_sys, reference, args.holo_ligand)
            onto = "the all-atom apo" if apo is not None else pdb.name
            print(f"holo {tag}, ligand {info['ligand']}: {info['atoms']} heavy atoms; holo superposed "
                  f"on {onto} ({info['paired']}/{info['matched']} alpha carbons kept, "
                  f"RMSD {info['fit_rmsd']:.2f} Å)")  # fmt: skip
            site_frequency(prefix, lig, args.view_dir, tag)
            holos.append((tag, moved, structure_format(path)))
            ligands.append((tag, lig))
        ranked = None
        if args.rank:
            crystal = crystal_pockets(apo, model, flags, where) if apo is not None else None
            from boonza.sites import particle_radii  # noqa: PLC0415

            # each bead's own radius (sigma/2) from the force field, for the pockets' enclosed cores
            bead_radii = particle_radii(system, np.asarray(ids, np.int64), "sigma")
            ranked = rank_trajectory(prefix, flags, crystal, ligands, where, args.consensus_cutoff,
                                     args.merge, args.merge_iso, bead_radii)
        if apo is not None:
            view = write_traj_view(prefix, apo, holos, args.holo_ligand, iso or 8.0, args.view_dir,
                                   structure_format(args.apo), ranked)
            print(f"\nall-atom view: cd {view.parent} && pymol {view.name}")
        else:
            print("\n(no all-atom view: give the all-atom apo structure with --apo)")


#: a consensus pocket: frames' pockets whose centres lie within this of its centroid
#: (--consensus-cutoff: larger merges neighbouring pockets, smaller splits a moving one).
#: 6 Å: on two sets of 40 apo trajectories it halves splitting, lifts Martini 3
#: Top-3/5/10 and leaves Martini 2, SIRAH and MOc within noise; 7-10 Å start to
#: merge neighbouring sites (benchmark: cutoff sweep, 2026-10-06)
CONSENSUS_CUTOFF = 6.0
#: cryptic: no pocket of the apo crystal structure within this of the consensus centre
CRYPTIC_CUTOFF = 4.0
#: quality is ranked first among pockets open in at least this share of frames
MIN_OCCUPANCY = 0.05
#: quality: this percentile of a pocket's per-frame probability over the frames it is open
QUALITY_PERCENTILE = 90


def pocket_blocks(info: Path) -> dict[int, str]:
    """fpocket's _info.txt cut into its pockets' blocks, by pocket number."""
    blocks, current, lines = {}, None, []
    for line in open(info):
        if line.startswith("Pocket "):
            if current is not None:
                blocks[current] = "".join(lines)
            current, lines = int(line.split()[1]), [line]
        elif current is not None:
            lines.append(line)
    if current is not None:
        blocks[current] = "".join(lines)
    return blocks


def run_fpocket_frame(pdb_text: str, xyz: np.ndarray, flags, workdir: Path):
    """fpocket on one frame: [(sphere centers, score, info block)] in fpocket's ranking."""
    lines = pdb_text.splitlines()
    out, k = [], 0
    for line in lines:
        if line.startswith("ATOM"):
            x, y, z = xyz[k]
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            k += 1
        out.append(line)
    frame = workdir / "frame.pdb"
    frame.write_text("\n".join(out) + "\n")
    shutil.rmtree(workdir / "frame_out", ignore_errors=True)
    subprocess.run([binary("fpocket"), "-f", str(frame), *flags],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    out_dir = workdir / "frame_out"
    if not (out_dir / "frame_pockets.pqr").exists():
        return []
    blocks = pocket_blocks(out_dir / "frame_info.txt")
    radii: dict[int, list] = {}
    for line in open(out_dir / "frame_pockets.pqr"):
        if line.startswith("ATOM"):
            radii.setdefault(int(line[22:26]), []).append(float(line[66:].split()[0]))
    return [(centers, np.array(radii.get(k, [])), score, blocks.get(k, "")) for k, (centers, score) in
            enumerate(read_pockets(out_dir, "frame"), 1)]  # fmt: skip


def burial(points: np.ndarray, protein: np.ndarray) -> float:
    """boonza.sites' burial (share of 26 directions that meet protein within 10 Å),
    averaged over a pocket's alpha-sphere centers."""
    import importlib  # noqa: PLC0415
    import types  # noqa: PLC0415

    occupancy = importlib.import_module("boonza.sites").Occupancy
    grid = types.SimpleNamespace(cell_centres=lambda: points)
    return float(occupancy.burial(grid, np.arange(len(points)), protein).mean())


def crystal_pockets(apo, model: str, flags, where: Path):
    """Pocket centres fpocket finds in the apo crystal structure, mapped to beads in the
    trajectory's frame: what is already open before the simulation."""
    import tempfile  # noqa: PLC0415

    cg = cgprep.coarse_grain(apo, model, f"protein and {cgprep.NOT_PROBES}")
    with tempfile.TemporaryDirectory() as tmp:
        pdb = Path(tmp) / "crystal.pdb"
        cgprep.write_fpocket_pdb(cg, pdb, model=kind(model), n_polar=n_polar(model))
        subprocess.run([binary("fpocket"), "-f", str(pdb), *flags],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
        out = pdb.with_name("crystal_out")
        if not (out / "crystal_pockets.pqr").exists():
            return np.zeros((0, 3))
        return np.array([c.mean(0) for c, _ in read_pockets(out, "crystal")])


def map_regions(freq_dx: Path, instances, iso: float):
    """Group pockets by the connected regions of mdpocket's frequency map at ``iso``.

    Each region (grid points at or above ``iso``, face-connected) is a consensus
    pocket; a frame's pocket joins the region holding most of its alpha-sphere
    centres, if at least a quarter of them lie in regions.  Returns (region groups,
    pockets in no region, to be grouped by centroid)."""
    from scipy import ndimage  # noqa: PLC0415

    grid, origin, delta = read_dx(freq_dx)
    labels, _ = ndimage.label(grid >= iso)
    step = np.diag(delta)
    groups, leftovers = {}, []
    for inst in instances:
        ijk = np.rint((inst["centers"] - origin) / step).astype(int)
        inside = np.all((ijk >= 0) & (ijk < np.array(grid.shape)), axis=1)
        lab = labels[tuple(ijk[inside].T)] if inside.any() else np.array([], int)
        lab = lab[lab > 0]
        if len(lab) < 0.25 * len(inst["centers"]):
            leftovers.append(inst)
            continue
        region = int(np.bincount(lab).argmax())
        g = groups.setdefault(region, {"sum": np.zeros(3), "count": 0, "members": [], "region": region})
        g["sum"] += inst["center"]
        g["count"] += 1
        g["members"].append(inst)
    return list(groups.values()), leftovers


def write_frame_pockets(path: Path, pockets) -> None:
    """Every per-frame fpocket pocket with the consensus pocket it joined, so rankings
    and merges can be re-derived without running fpocket again.

    Arrays (one row per per-frame pocket): ``consensus`` (its consensus pocket's
    quality rank), ``frame``, ``rank`` (fpocket's rank in that frame), ``p``,
    ``score``, ``burial``, ``center`` (n, 3), ``block`` (fpocket's info text); its alpha
    spheres are ``sphere_centers``/``sphere_radii`` rows ``offsets[i]:offsets[i + 1]``.
    """
    rows = [(q["rank_quality"], m) for q in pockets for m in q["members"]]
    spheres = [m["centers"] for _, m in rows]
    np.savez_compressed(
        path,
        consensus=np.array([c for c, _ in rows], int),
        frame=np.array([m["frame"] for _, m in rows], int),
        rank=np.array([m["rank"] for _, m in rows], int),
        p=np.array([m["p"] for _, m in rows]),
        score=np.array([m["score"] for _, m in rows]),
        burial=np.array([m["burial"] for _, m in rows]),
        center=np.array([m["center"] for _, m in rows]).reshape(-1, 3),
        offsets=np.cumsum([0] + [len(c) for c in spheres]),
        sphere_centers=np.vstack(spheres) if spheres else np.zeros((0, 3)),
        sphere_radii=np.concatenate([m["radii"] for _, m in rows]) if rows else np.zeros(0),
        block=np.array([m["block"] for _, m in rows], dtype=str),
    )


def rank_trajectory(prefix: Path, flags, crystal, ligands, where: Path,
                    cutoff: float = CONSENSUS_CUTOFF, merge: str = "centroid",
                    merge_iso: float = 0.2, bead_radii=None):  # fmt: skip
    """Consensus pockets over a trajectory, ranked three ways.

    fpocket (with the model's preset) runs on every frame of ``prefix``.dcd;
    each pocket's score becomes a probability (the refitted score is a
    logistic model) and gets boonza.sites' burial.  Pockets of all frames are
    grouped greedily, best first: a pocket joins the consensus pocket whose
    centroid is within ``cutoff`` of its centre, or starts one
    (``merge="centroid"``).  With ``merge="map"`` the consensus pockets are the
    regions of mdpocket's frequency map at ``merge_iso`` (:func:`map_regions`),
    a region's pieces in one frame taken together, and only pockets outside every
    region are grouped by centroid.
    Each frame counts once per consensus pocket, with its best member.

    - persistence: mean probability over all frames (0 where it is absent);
    - quality: the ``QUALITY_PERCENTILE``th percentile of its probability over
      the frames it is open in; pockets open in fewer than ``MIN_OCCUPANCY`` of
      the frames rank after the others;
    - quality x buriedness: quality times its mean burial over those frames.

    A consensus pocket with no pocket of the apo crystal structure (``crystal``)
    within ``CRYPTIC_CUTOFF`` is flagged cryptic.  With ``bead_radii`` (one per bead
    of ``prefix``.pdb), each consensus pocket also gets its enclosed core on its best
    frame (core.py: SiteMap's site-point rules on the beads), a compact,
    ligand-sized site; the ranking is not changed by it.  ``ligands``: [(tag, heavy
    atoms)] to score each pocket against.  Writes consensus_pockets.csv,
    pockets_vs_holo[_tag].csv, fpocket_info.txt, frames.csv, frame_pockets.npz,
    consensus_pockets.pqr and consensus_cores.pqr into ``where``; returns the
    consensus pockets.
    """
    import importlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import boonza  # noqa: PLC0415

    sites = importlib.import_module("boonza.sites")
    beads = load(Path(f"{prefix}.pdb"))
    traj = boonza.open_trajectory(f"{prefix}.dcd", beads)
    pdb_text = Path(f"{prefix}.pdb").read_text()
    instances, frames, coords = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for f, frame in enumerate(traj):
            xyz = np.asarray(frame.positions, float)
            coords.append(xyz.astype(np.float32))
            pockets = run_fpocket_frame(pdb_text, xyz, flags, Path(tmp))
            frames.append(len(pockets))
            for rank, (centers, radii, score, block) in enumerate(pockets, 1):
                p = float(1.0 / (1.0 + np.exp(-score)))
                instances.append({"frame": f, "rank": rank, "p": p, "score": score, "centers": centers,
                                  "radii": radii,
                                  "center": centers.mean(0), "burial": burial(centers, xyz),
                                  "block": block})  # fmt: skip
    n = len(frames)
    groups = []  # each: sum of centres, count, members
    leftovers = sorted(instances, key=lambda d: -d["p"])
    if merge == "map":  # consensus pockets = regions of mdpocket's frequency map
        groups, leftovers = map_regions(Path(f"{prefix}_freq.dx"), leftovers, merge_iso)
    for inst in leftovers:  # by centroid: every pocket (centroid merge), or those outside regions
        best, dist = None, cutoff
        for g in groups:
            d = float(np.linalg.norm(g["sum"] / g["count"] - inst["center"]))
            if d < dist:
                best, dist = g, d
        if best is None:
            best = {"sum": np.zeros(3), "count": 0, "members": []}
            groups.append(best)
        best["sum"] += inst["center"]
        best["count"] += 1
        best["members"].append(inst)
    pockets = []
    for g in groups:
        per_frame = {}
        for m in g["members"]:  # best member per frame (members come best first)
            per_frame.setdefault(m["frame"], m)
        if g.get("region"):  # a region's pieces in a frame are one pocket: their spheres together
            pieces = defaultdict(list)
            for m in g["members"]:
                pieces[m["frame"]].append(m)
            for f, best in per_frame.items():
                if len(pieces[f]) > 1:
                    centers = np.vstack([m["centers"] for m in pieces[f]])
                    per_frame[f] = {**best, "centers": centers, "center": centers.mean(0),
                                    "radii": np.concatenate([m["radii"] for m in pieces[f]])}  # fmt: skip
        open_ = list(per_frame.values())
        p_open = np.array([m["p"] for m in open_])
        occupancy = len(open_) / n
        quality = float(np.percentile(p_open, QUALITY_PERCENTILE))
        buried = float(np.mean([m["burial"] for m in open_]))
        rep = max(open_, key=lambda m: (m["p"], -m["frame"]))
        volume = re.search(r"Volume :\s*([-\d.]+)", rep["block"])
        center = np.mean([m["center"] for m in open_], axis=0)
        cryptic = None if crystal is None else (
            not len(crystal) or float(np.linalg.norm(crystal - center, axis=1).min()) >= CRYPTIC_CUTOFF)
        pockets.append({"occupancy": occupancy, "persistence": float(p_open.sum() / n),
                        "quality": quality, "burial": buried, "quality_burial": quality * buried,
                        "center": center, "rep": rep, "cryptic": cryptic, "frames_open": len(open_),
                        "volume": float(volume.group(1)) if volume else float("nan"),
                        "alpha_spheres": len(rep["centers"]),
                        "open": open_, "members": g["members"]})  # fmt: skip
    for key in ("persistence", "quality", "quality_burial"):
        floor = key != "persistence"
        order = sorted(range(len(pockets)), key=lambda i: (
            floor and pockets[i]["occupancy"] < MIN_OCCUPANCY, -pockets[i][key]))  # fmt: skip
        for r, i in enumerate(order, 1):
            pockets[i][f"rank_{key}"] = r
    pockets.sort(key=lambda q: q["rank_quality"])
    for q in pockets:
        q["core"] = (core.core_points(q["rep"]["centers"], q["rep"]["radii"], coords[q["rep"]["frame"]],
                                      bead_radii) if bead_radii is not None else np.zeros((0, 3)))  # fmt: skip
    where.mkdir(parents=True, exist_ok=True)
    head = ["rank_quality", "rank_persistence", "rank_quality_burial", "quality", "persistence",
            "quality_burial", "occupancy", "burial", "cryptic", "frames_open", "best_frame",
            "volume_best_frame", "alpha_spheres_best_frame",
            "center_x", "center_y", "center_z",
            "core_volume", "core_center_x", "core_center_y", "core_center_z"]  # fmt: skip

    def base(q):
        c = q["center"]
        return [q["rank_quality"], q["rank_persistence"], q["rank_quality_burial"], f"{q['quality']:.3f}",
                f"{q['persistence']:.3f}", f"{q['quality_burial']:.3f}", f"{q['occupancy']:.3f}",
                f"{q['burial']:.3f}", "" if q["cryptic"] is None else q["cryptic"], q["frames_open"],
                q["rep"]["frame"], f"{q['volume']:.1f}", q["alpha_spheres"],
                f"{c[0]:.3f}", f"{c[1]:.3f}", f"{c[2]:.3f}",
                f"{len(q['core']) * core.SPACING**3:.0f}",
                *([f"{v:.3f}" for v in q["core"].mean(0)] if len(q["core"]) else ["", "", ""])]  # fmt: skip

    write_frame_pockets(where / "frame_pockets.npz", pockets)
    with open(where / "consensus_pockets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(base(q) for q in pockets)
    for tag, lig in ligands:
        name = "pockets_vs_holo.csv" if len(ligands) == 1 else f"pockets_vs_holo_{tag}.csv"
        with open(where / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head + ["center_to_nearest_ligand_atom", "center_to_ligand_centroid",
                               "ligand_atoms_within_3A", "spheres_within_3A", "PPc", "MOc",
                               "share_of_open_frames_PPc", "core_center_to_nearest_ligand_atom", "PPc_core",
                               "core_ligand_volume_covered", "core_volume_near_ligand",
                               "core_volume_in_ligand", "core_DVO"])  # fmt: skip
            for q in pockets:
                spheres = q["rep"]["centers"]
                near = np.linalg.norm(lig[:, None] - spheres[None], axis=2) < MOC_D
                lig_cov, pocket_cov = float(near.any(1).mean()), float(near.any(0).mean())
                dca = sites.dca(q["center"][None], lig)
                dcc = sites.dcc(q["center"][None], lig)
                # in how many of the frames it is open its centre there passes PPc
                hits = sum(sites.dca(m["center"][None], lig) < PPC_CUTOFF for m in q["open"])
                q.setdefault("verdicts", []).append((dca < PPC_CUTOFF,
                                                     lig_cov > MOC_LIGAND and pocket_cov > MOC_POCKET))
                if len(q["core"]):
                    core_dca = float(np.linalg.norm(lig - q["core"].mean(0), axis=1).min())
                    ov = voxels.overlap(voxels.grid_pocket(q["core"], core.SPACING), lig)
                    core_cols = [f"{core_dca:.3f}", core_dca < PPC_CUTOFF, f"{ov['ligand_volume_covered']:.3f}",
                                 f"{ov['pocket_volume_near_ligand']:.3f}", f"{ov['pocket_volume_in_ligand']:.3f}",
                                 f"{ov['DVO']:.3f}"]  # fmt: skip
                else:
                    core_cols = ["", False, "", "", "", ""]
                w.writerow(base(q) + [f"{dca:.3f}", f"{dcc:.3f}", f"{lig_cov:.3f}", f"{pocket_cov:.3f}",
                                      dca < PPC_CUTOFF, lig_cov > MOC_LIGAND and pocket_cov > MOC_POCKET,
                                      f"{hits / len(q['open']):.3f}", *core_cols])  # fmt: skip
    with open(where / "fpocket_info.txt", "w") as fh:
        for q in pockets:
            r = q["rep"]
            fh.write(f"Consensus pocket {q['rank_quality']} (quality rank; persistence rank "
                     f"{q['rank_persistence']}, quality x burial rank {q['rank_quality_burial']}): "
                     f"best in frame {r['frame']}, where it is fpocket's pocket {r['rank']}\n")
            fh.write(r["block"].split("\n", 1)[1] if "\n" in r["block"] else "\n")
    with open(where / "frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "pockets"])
        w.writerows(enumerate(frames))
    with open(where / "consensus_pockets.pqr", "w") as fh:
        k = 0
        for q in pockets:
            for (x, y, z), r in zip(q["rep"]["centers"], q["rep"]["radii"], strict=True):
                k += 1  # as fpocket writes its pockets: the alpha sphere's radius in the last column
                fh.write(f"ATOM  {k % 100000:5d}    C STP  {q['rank_quality'] % 10000:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  0.00 {r:6.2f}\n")
        fh.write("END\n")
    with open(where / "consensus_cores.pqr", "w") as fh:
        k, r = 0, voxels.cell_radius(core.SPACING)
        for q in pockets:
            for x, y, z in q["core"]:
                k += 1  # one grid cell of the core per point, its radius that of a sphere of equal volume
                fh.write(f"ATOM  {k % 100000:5d}    C COR  {q['rank_quality'] % 10000:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  0.00 {r:6.2f}\n")
        fh.write("END\n")
    print(f"ranked {len(pockets)} consensus pockets from {len(instances)} pockets in {n} frames "
          f"-> {where / 'consensus_pockets.csv'}")  # fmt: skip
    return pockets


def site_frequency(prefix: Path, lig: np.ndarray, view_dir=None, tag: str = "holo") -> None:
    """How often, and how densely, the holo ligand's site is a pocket over the trajectory.

    For each ligand heavy atom: the largest pocket frequency (fraction of
    frames) and alpha-sphere density within 2 Å of it in mdpocket's maps."""
    freq, f_origin, f_delta = read_dx(Path(f"{prefix}_freq.dx"))
    dens, d_origin, d_delta = read_dx(Path(f"{prefix}_dens.dx"))

    def around(grid, origin, delta, xyz, reach=2.0):
        step = np.diag(delta)
        lo = np.floor((xyz - reach - origin) / step).astype(int).clip(0, np.array(grid.shape) - 1)
        hi = np.ceil((xyz + reach - origin) / step).astype(int).clip(0, np.array(grid.shape) - 1)
        return float(grid[lo[0]: hi[0] + 1, lo[1]: hi[1] + 1, lo[2]: hi[2] + 1].max())

    f = np.array([around(freq, f_origin, f_delta, x) for x in lig])
    d = np.array([around(dens, d_origin, d_delta, x) for x in lig])
    where = Path(view_dir) if view_dir else Path(f"{prefix}").parent
    where.mkdir(parents=True, exist_ok=True)
    table = where / f"ligand_site_frequency_{tag}.csv"
    with open(table, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["ligand_atom", "x", "y", "z", "pocket_frequency_within_2A", "density_within_2A"])
        for k, (x, fk, dk) in enumerate(zip(lig, f, d, strict=True), 1):
            out.writerow([k, f"{x[0]:.3f}", f"{x[1]:.3f}", f"{x[2]:.3f}", f"{fk:.3f}", f"{dk:.2f}"])
    print(f"ligand site of {tag} over the trajectory: pocket frequency median {np.median(f):.2f} "
          f"(max within 2 Å of each ligand atom); {np.mean(f >= 0.5):.0%} of ligand atoms in a pocket "
          f"in at least half the frames -> {table}")  # fmt: skip


#: the view shows the consensus pockets in the top this-many of any of the three rankings
VIEW_TOP = 5


def verdicts(q) -> list:
    """(PPc, MOc) of a consensus pocket against each holo ligand: from a ranked pocket,
    or from a row of pockets_vs_holo.csv."""
    if "verdicts" in q:
        return q["verdicts"]
    if "PPc" in q:
        return [(str(q["PPc"]) == "True", str(q["MOc"]) == "True")]
    return []


def view_selection(ranked) -> list:
    """The consensus pockets drawn in the view: those in the top ``VIEW_TOP`` of any
    ranking, and the best-ranked (by quality) one correct by PPc and by MOc for the holo
    ligand, whatever its rank -- so the site is never hidden, and a site split into many
    small consensus pockets adds at most two."""
    top = {id(q) for q in ranked if min(int(q["rank_quality"]), int(q["rank_persistence"]),
                                        int(q["rank_quality_burial"])) <= VIEW_TOP}  # fmt: skip
    for k in (0, 1):  # PPc, MOc
        hits = [q for q in ranked if verdicts(q) and verdicts(q)[0][k]]
        if hits:
            top.add(id(min(hits, key=lambda q: int(q["rank_quality"]))))
    return [q for q in ranked if id(q) in top]


def consensus_view_lines(ranked) -> list[str]:
    """PyMOL lines drawing the consensus pockets as the single-frame view draws pockets.

    Each pocket's alpha spheres in its best frame, with their radii, as its own
    object ``pocket_<quality rank>`` (see :func:`pocket_object_lines`),
    labelled with that rank, * when cryptic, and PPc / MOc when correct for the holo
    ligand.  Only the pockets of :func:`view_selection` are drawn; every pocket is in
    the CSV files.  ``ranked``: ranked pockets, or rows of pockets_vs_holo.csv."""
    drawn = []
    for q in view_selection(ranked):
        if "center" in q:  # a ranked pocket, or a row of consensus_pockets.csv
            centre = tuple(float(v) for v in q["center"])
        else:
            centre = tuple(float(q[f"center_{a}"]) for a in "xyz")
        tag = f"{int(q['rank_quality'])}{'*' if str(q['cryptic']) == 'True' else ''}"
        for ppc, moc in verdicts(q)[:1]:
            tag += (" PPc" if ppc else "") + (" MOc" if moc else "")
        drawn.append((int(q["rank_quality"]), tag, centre))
    lines = ["# consensus pockets: the top 5 of any ranking, and the best-ranked one correct for the",
             "# holo ligand by PPc and by MOc, drawn in their best frame. Each is the object",
             "# pocket_<quality rank> (its alpha spheres and its label): click it in the object panel to",
             "# show or hide it. Label = quality rank, * = cryptic (closed in the apo crystal), PPc/MOc =",
             "# correct for the holo ligand. All ranks and numbers are in pockets_vs_holo.csv",
             "load consensus_pockets.pqr, consensus_all"]  # fmt: skip
    lines += pocket_object_lines("consensus_all", drawn)
    lines += ["# each pocket's enclosed core (SiteMap's site-point rules on the beads, best frame):",
              "# the object core_<quality rank>, one sphere per 2 A grid cell",
              "load consensus_cores.pqr, cores_all"]  # fmt: skip
    for k, _, _ in drawn:
        color = RANK_COLORS[k - 1] if 1 <= k <= len(RANK_COLORS) else "grey50"
        lines += [f"create core_{k}, cores_all and resi {k}", f"hide everything, core_{k}",
                  f"show spheres, core_{k}", f"color {color}, core_{k}",
                  f"set sphere_transparency, 0.45, core_{k}"]  # fmt: skip
    lines += ["delete cores_all", "# end of consensus pockets"]
    return lines


def write_traj_view(prefix: Path, apo, holos=(), ligand: str | None = None, level: float = 8.0,
                    view_dir=None, fmt: str = "pdb", ranked=None) -> Path:  # fmt: skip
    """A PyMOL script with mdpocket's maps on the all-atom structures, beads hidden.

    The apo protein as a translucent wheat cartoon; the pocket frequency map as
    a translucent red surface at 0.5 (a pocket in at least half the frames);
    the alpha-sphere density as a yellow mesh at the model's calibrated level;
    each holo structure ((tag, structure, format), in the frame) adds its ligand
    as blue sticks and its protein (light blue) loaded but hidden."""
    import shutil as sh  # noqa: PLC0415

    from boonza.io import save as save_structure  # noqa: PLC0415

    where = Path(view_dir) if view_dir else Path(f"{prefix}").parent
    where.mkdir(parents=True, exist_ok=True)
    sh.copy(f"{prefix}_freq.dx", where / "pocket_frequency.dx")
    sh.copy(f"{prefix}_dens.dx", where / "pocket_density.dx")
    save_structure(apo, where / f"apo.{fmt}")
    lines = ["# mdpocket maps of a coarse-grained trajectory, on the all-atom structures (cgpocket)",
             f"load apo.{fmt}, apo"]  # fmt: skip
    for tag, holo, hf in holos:
        save_structure(holo.select(f"not ({ligand})").clone(), where / f"holo_{tag}.{hf}")
        save_structure(holo.select(f"({ligand})").clone(), where / f"ligand_{tag}.{hf}")
        lines += [f"load holo_{tag}.{hf}, holo_{tag}", f"load ligand_{tag}.{hf}, ligand_{tag}"]
    lines += ["load pocket_frequency.dx, frequency", "load pocket_density.dx, density",
              "hide everything", "bg_color black", "set ray_opaque_background, 0",
              "show cartoon, apo", "color wheat, apo", "set cartoon_transparency, 0.45, apo",
              "# pockets open in at least half the frames (isolevel pocket_frequency, 0.3 shows rarer ones)",
              "isosurface pocket_frequency, frequency, 0.5",
              "color red, pocket_frequency", "set transparency, 0.4, pocket_frequency",
              "# the maps are off by default: click pocket_frequency / pocket_density to show them",
              "# alpha-sphere density at the model's calibrated level (mdpocket's 8 is for all-atom)",
              f"isomesh pocket_density, density, {level:g}",
              "color yellow, pocket_density", "disable pocket_frequency", "disable pocket_density"]  # fmt: skip
    if ranked:
        lines += consensus_view_lines(ranked)
    if holos:
        lines += ["show cartoon, holo_*", "color lightblue, holo_*",
                  "# click a holo_ object in the object panel to compare the folds", "disable holo_*",
                  "show sticks, ligand_* and not hydro", "color tv_blue, ligand_* and elem C",
                  "set stick_radius, 0.3, ligand_*", "zoom ligand_*, 10"]  # fmt: skip
    else:
        lines.append("zoom pocket_frequency")
    view = where / "view.pml"
    view.write_text("\n".join(lines) + "\n")
    return view


def read_dx(dx: Path):
    """(grid values, origin, the three delta vectors) of a .dx map."""
    values, counts, origin, delta = [], None, None, []
    for line in open(dx):
        w = line.split()
        if not w or w[0].startswith("#"):
            continue
        if line.startswith("object 1"):
            counts = tuple(int(x) for x in w[-3:])
        elif w[0] == "origin":
            origin = np.array(w[1:4], float)
        elif w[0] == "delta":
            delta.append(np.array(w[1:4], float))
        elif w[0][0].isdigit() or w[0][0] in "-.":
            values += [float(x) for x in w]
    return np.array(values[: int(np.prod(counts))]).reshape(counts), origin, np.array(delta)


def write_isosurface(dx: Path, level: float) -> Path:
    """Grid points of an mdpocket .dx map at or above ``level``, as mdpocket writes them."""
    grid, origin, delta = read_dx(dx)
    out = dx.with_name(dx.name.replace("_dens.dx", f"_dens_iso_{level:g}.pdb"))
    lines = []
    for k, (i, j, l) in enumerate(np.argwhere(grid >= level)):
        x, y, z = origin + i * delta[0] + j * delta[1] + l * delta[2]
        lines.append(f"ATOM  {(k + 1) % 100000:5d}  C   PTH     1    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"{0.0:6.2f}{grid[i, j, l]:6.2f}")  # density in the B-factor column
    out.write_text("\n".join(lines) + "\nEND\n")
    return out


def cmd_flags(args, extra):
    print(" ".join(merge_flags(preset(args.model), extra)))


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1 :]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    models = ["aa", "martini2", "martini3", "sirah"]

    for name, fn, helptext in (("prep", cmd_prep, "write an fpocket-ready PDB"),
                               ("run", cmd_run, "prep, then fpocket with the preset")):  # fmt: skip
        p = sub.add_parser(name, help=helptext)
        p.add_argument("input", help="structure: .mae .pdb .cif .gro .dms ... (all-atom or CG)")
        p.add_argument("--model", required=True, choices=models)
        p.add_argument("--top", help="GROMACS topology of a CG input (gives the bead types)")
        p.add_argument("--selection", help="atoms to keep (default: protein)")
        p.add_argument("-o", "--output", default=".", help="a .pdb, or a directory")
        if name == "run":
            p.add_argument("--holo", help="holo structure: score each pocket against its ligand")
            p.add_argument("--holo-ligand", default="resname LIG",
                           help='the ligand in --holo (default: "resname LIG")')  # fmt: skip
            p.add_argument("--holo-top", help="topology of a coarse-grained --holo")
            p.add_argument("--apo", help="the all-atom apo structure, for the view (default: the "
                           "input, when it is all-atom); a CG input is superposed on it")  # fmt: skip
            p.add_argument("--view-dir", help="write the PyMOL view and its structures there "
                           "(view.pml, apo/holo/ligand in MAE when the inputs are MAE)")  # fmt: skip
        p.set_defaults(fn=fn)

    p = sub.add_parser("traj", help="CG trajectory -> protein-only PDB + DCD (+ mdpocket)")
    p.add_argument("--coords", required=True, help="CG structure (.gro, .pdb, .dms) of the trajectory")
    p.add_argument("--top", help="GROMACS topology, for bead types and bonds")
    p.add_argument("--traj", required=True, help=".xtc .trr .dcd .nc .dtr")
    p.add_argument("--model", required=True, choices=models[1:])
    p.add_argument("--selection", help="beads to keep (default: protein)")
    p.add_argument("--fit", help="beads to fit every frame on (default: the protein backbone, "
                   "BB / GC / CA); chain LIG probes are never used")  # fmt: skip
    p.add_argument("--no-fix", action="store_true", help="do not make whole and fit")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("-o", "--output", required=True, help="output prefix")
    p.add_argument("--run", action="store_true", help="run mdpocket as well")
    p.add_argument("--apo", help="the all-atom apo structure: an all-atom view of the maps "
                   "(superposed on the trajectory's reference frame)")  # fmt: skip
    p.add_argument("--holo", nargs="+", help="holo structure(s): each one's ligand in the view, "
                   "and how often its site is a pocket over the trajectory")  # fmt: skip
    p.add_argument("--holo-ligand", default="resname LIG",
                   help='the ligand in --holo (default: "resname LIG")')  # fmt: skip
    p.add_argument("--holo-top", help="topology of a coarse-grained --holo")
    p.add_argument("--view-dir", help="write the view and its files there (default: beside --output)")
    p.add_argument("--rank", action="store_true", help="also run fpocket on every (strided) frame and "
                   "rank consensus pockets by persistence, quality and quality x buriedness")  # fmt: skip
    p.add_argument("--merge", choices=("centroid", "map"), default="centroid",
                   help="how pockets of different frames become one consensus pocket: by centroid "
                   "distance (--consensus-cutoff), or by region of the pocket-frequency map "
                   "(--merge-iso)")  # fmt: skip
    p.add_argument("--merge-iso", type=float, default=0.2,
                   help="frequency-map level whose connected regions are the consensus pockets with "
                   "--merge map (lower merges more, higher splits more; default 0.2)")  # fmt: skip
    p.add_argument("--consensus-cutoff", type=float, default=CONSENSUS_CUTOFF,
                   help="Å: a frame's pocket joins a consensus pocket whose centroid is this close "
                   "(larger merges neighbouring pockets, smaller splits a moving one; default 6)")  # fmt: skip
    p.set_defaults(fn=cmd_traj)

    p = sub.add_parser("flags", help="print the tuned fpocket/mdpocket flags")
    p.add_argument("--model", required=True, choices=models)
    p.set_defaults(fn=cmd_flags)

    args = ap.parse_args(argv)
    args.fn(args, extra)


if __name__ == "__main__":
    main()
