"""Turn all-atom or coarse-grained structures into PDB files fpocket can read.

fpocket knows two things about an atom: where it is, and its element.  The
element decides whether the atom is apolar (Pauling electronegativity < 2.8,
i.e. C or S) and hydrogens are dropped before the Voronoi tessellation.  A
coarse-grained bead has no element, so this module writes one that carries
the bead's polarity instead:

* apolar beads are written as ``C``, polar beads as ``O``;
* Martini beads are classed by their type (C and X apolar; P, Q and D polar;
  the intermediate N class is a choice, ``n_polar``);
* SIRAH backbone beads go by name (GN and GO polar, GC apolar); a SIRAH
  side-chain bead is polar when it carries a partial charge (|q| >= 0.1).
  The name alone misleads there: lysine's BCG and BCE, arginine's BCZ, the
  carboxylate carbons of Asp and Glu and tyrosine's hydroxyl beads are named
  after carbons but are charged.  The charges are SIRAH's own, from boonza's
  ``sirahize``, or from the topology the beads were loaded with.

Residue names are written as the standard three-letter amino acids
(SIRAH's ``sA``, ``sHe`` ...; Martini's ``HSD``, ``CYX`` ...), which fpocket
uses for its polarity score.

The Martini and SIRAH mappings themselves are boonza's
(``boonza.martinize``, ``boonza.sirah.sirahize``).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np

MODELS = ("aa", "martini2", "martini3", "sirah")
FORCEFIELDS = {"martini2": "martini22", "martini3": "martini3001"}

APOLAR, POLAR = "C", "O"

#: the chain probes (ligand copies placed around the protein, e.g. by boonza swim)
#: are written to: they are never part of the protein a pocket is searched in
PROBE_CHAIN = "LIG"
NOT_PROBES = f'not chain "{PROBE_CHAIN}"'


def probe_ids(system) -> np.ndarray:
    """Atoms (beads) of the probe chain."""
    chain_of = np.asarray(system.residues["chain"])[np.asarray(system.atoms["residue"])]
    names = np.array([str(c).strip() for c in system.chains["name"]])
    return np.flatnonzero(names[chain_of] == PROBE_CHAIN) if len(names) else np.array([], int)

_SIRAH_RESNAMES = {
    "sA": "ALA", "sR": "ARG", "sN": "ASN", "sD": "ASP", "sC": "CYS", "sX": "CYS",
    "sQ": "GLN", "sE": "GLU", "sG": "GLY", "sHe": "HIS", "sHd": "HIS", "sHp": "HIS",
    "sI": "ILE", "sL": "LEU", "sK": "LYS", "sM": "MET", "sF": "PHE", "sP": "PRO",
    "sS": "SER", "sT": "THR", "sW": "TRP", "sY": "TYR", "sV": "VAL",
}  # fmt: skip
_STANDARD_RESNAMES = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
}  # fmt: skip


def martini_polar(bead_type: str, n_polar: bool = False) -> bool:
    """Whether a Martini 2 or 3 bead type is polar.

    The size prefix (S, T), Martini 2's amino-acid prefix (A: the AC1/AC2
    side chains of Leu, Ile and Val) and the label suffixes (d, a, h, e, r, q,
    p, n, ...) are ignored; the class letter decides.
    """
    m = re.match(r"^(?:[ST]|A(?=C))?([PNCQXD])", str(bead_type))
    if m is None:
        return True  # an unknown bead is assumed to be polar
    cls = m.group(1)
    if cls == "N":
        return n_polar
    return cls not in "CX"


#: a SIRAH side-chain bead with at least this much partial charge is polar
SIRAH_CHARGED = 0.1


def sirah_polar(bead_name: str, charge: float) -> bool:
    """Whether a SIRAH bead is polar: the backbone by name, a side chain by its charge."""
    name = str(bead_name).strip().upper()
    if name in ("GN", "GO"):
        return True
    if name == "GC":
        return False
    return abs(charge) >= SIRAH_CHARGED - 1e-6


def sirah_charges(names, resnames) -> list[float]:
    """SIRAH's partial charge of each bead, from its residue library."""
    from boonza.sirah.build import read_residues  # noqa: PLC0415

    library = read_residues()[0]
    charges = []
    for name, resname in zip(names, resnames, strict=True):
        entry = library.get(str(resname).strip())
        by_name = {n: q for n, _kind, q in entry.atoms} if entry else {}
        charges.append(float(by_name.get(str(name).strip(), 0.0)))
    return charges


def standard_resname(resname: str) -> str:
    resname = str(resname).strip()
    if resname in _SIRAH_RESNAMES:
        return _SIRAH_RESNAMES[resname]
    return _STANDARD_RESNAMES.get(resname.upper(), resname.upper()[:3])


def guess_model(names, resnames) -> str:
    """``sirah``, ``martini`` or ``aa`` from bead names and residue names."""
    names = {str(n).strip() for n in names}
    if {"GN", "GC", "GO"} & names and any(str(r).startswith("s") for r in resnames):
        return "sirah"
    if "BB" in names:
        return "martini"
    return "aa"


def write_pdb(path, names, resnames, chains, resids, insertions, positions, elements):
    """A plain PDB of ATOM records with the given elements (columns 77-78)."""
    lines = []
    for k, (name, resname, chain, resid, ins, xyz, el) in enumerate(
        zip(names, resnames, chains, resids, insertions, positions, elements, strict=True)
    ):
        name = str(name)[:4]
        name = f" {name:<3s}" if len(name) < 4 else name
        lines.append(
            f"ATOM  {(k + 1) % 100000:5d} {name:4s} {str(resname)[:3]:>3s} "
            f"{(str(chain) or 'A')[:1]:1s}{int(resid) % 10000:4d}{(str(ins) or ' ')[:1]:1s}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2s}"
        )
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def _residue_columns(system, ids):
    res = np.asarray(system.atoms["residue"])[ids]
    resnames = np.asarray(system.residues["name"])[res]
    resids = np.asarray(system.residues["resid"])[res]
    ins = np.asarray(system.residues["insertion"])[res]
    chain_of = np.asarray(system.residues["chain"])[res]
    chains = np.asarray(system.chains["name"])[chain_of]
    return resnames, resids, ins, chains


def bead_elements(system, ids=None, model: str | None = None, n_polar: bool = False):
    """The element to write for each atom or bead: real for all-atom, C/O for beads."""
    ids = np.arange(system.natoms) if ids is None else np.asarray(ids)
    names = np.asarray(system.atoms["name"])[ids]
    resnames = _residue_columns(system, ids)[0]
    model = model or guess_model(names, resnames)
    if model == "aa":
        from boonza.elements import symbol  # noqa: PLC0415

        return [symbol(int(a)) for a in np.asarray(system.atoms["anum"])[ids]]
    if model == "sirah":
        charges = np.asarray(system.atoms["charge"])[ids]
        if not np.any(charges):  # loaded without its topology: take the library's
            charges = sirah_charges(names, resnames)
        return [POLAR if sirah_polar(n, q) else APOLAR for n, q in zip(names, charges, strict=True)]
    types = np.asarray(system.atoms["type"])[ids]
    if not any(str(t) for t in types):
        raise ValueError(
            "the Martini beads have no types: load the structure with its topology "
            "(.top/.itp), or pass a bead-type table"
        )
    return [POLAR if martini_polar(t, n_polar) else APOLAR for t in types]


def write_fpocket_pdb(system, path, ids=None, model: str | None = None, n_polar: bool = False):
    """Write ``system`` (or the atoms ``ids``) as a PDB for fpocket/mdpocket."""
    ids = np.arange(system.natoms) if ids is None else np.asarray(ids)
    names = np.asarray(system.atoms["name"])[ids]
    resnames, resids, ins, chains = _residue_columns(system, ids)
    elements = bead_elements(system, ids, model, n_polar)
    write_pdb(path, names, [standard_resname(r) for r in resnames], chains, resids, ins,
              np.asarray(system.positions)[ids], elements)  # fmt: skip


def coarse_grain(system, model: str, atoms: str = f"protein and {NOT_PROBES}",
                 drop_unknown: bool = True):  # fmt: skip
    """``atoms`` of an all-atom ``system`` mapped onto ``model`` beads, as a boonza System.

    ``aa`` returns the heavy atoms themselves.  The CG system keeps the chain
    names and residue numbers of the all-atom one; probes (chain LIG) are not
    protein.  ``drop_unknown`` leaves
    out, with a warning, residues the model has no mapping for (a D-amino
    acid, a modified residue) rather than refusing the structure.
    """
    if model == "aa":
        return system.select(f"({atoms}) and not hydrogen").clone()
    dropped: list[str] = []
    while True:
        names = " ".join(f'"{d}"' for d in dropped)  # quoted: a residue name can be numeric
        selection = atoms if not dropped else f"({atoms}) and not resname {names}"
        try:
            cg = _map(system, model, selection)
            break
        except ValueError as e:
            unknown = _unknown_residues(str(e))
            if not drop_unknown or not unknown or set(unknown) <= set(dropped):
                raise
            dropped += [u for u in unknown if u not in dropped]
    if dropped:
        warnings.warn(f"{model}: left out residues it cannot map: {' '.join(dropped)}", stacklevel=2)
    if model != "sirah":  # SIRAH beads carry their residues' numbers already
        _renumber_like(cg, system, selection)
    return cg


def _map(system, model, atoms):
    import boonza  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model in FORCEFIELDS:
            built = boonza.martinize(system, atoms, forcefield=FORCEFIELDS[model], cofactors=False)
        elif model == "sirah":
            return _sirah_beads(system, atoms)
        else:
            raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
        return built.system()


def _sirah_beads(system, atoms):
    """SIRAH beads as a System, straight from SIRAH's map, with SIRAH's charges.

    The beads and residues come from the map, not from the built topology:
    the topology keys beads by residue number alone, so its residues cannot
    carry insertion codes (77, 77A), which crystal structures of serine
    proteases and others are full of.  ``sirahize`` is run only for the
    partial charges, which it assigns bead by bead.
    """
    import boonza  # noqa: PLC0415
    from boonza import sirah  # noqa: PLC0415

    beads = sirah.map_structure(system, atoms)
    # SIRAH's partial charges, which decide the polarity of side-chain beads
    try:
        built = sirah.sirahize(system, atoms, termini="Neutral")
        charge = {(str(b.chain).strip(), int(b.resid), str(b.insertion).strip(), b.name): q
                  for mol in built.molecules for b, q in zip(mol.beads, mol.charges, strict=True)}  # fmt: skip
    except (IndexError, KeyError, ValueError):
        # sirahize fails on some chains (1IGJ); the residue library gives the
        # same side-chain charges, and backbone polarity goes by name
        lib = sirah_charges([b.name for b in beads], [b.residue for b in beads])
        charge = {(str(b.chain).strip(), int(b.resid), str(b.insertion).strip(), b.name): q
                  for b, q in zip(beads, lib, strict=True)}  # fmt: skip
    s = boonza.System("sirah")
    chain, residue, key = None, None, None
    for b in beads:
        if chain is None or b.chain != s.chains["name"][chain.id]:
            chain = s.add_chain(name=b.chain)
            key = None
        if (b.resid, b.insertion, b.residue) != key:
            residue = s.add_residue(chain, name=b.residue, resid=int(b.resid), insertion=b.insertion)
            key = (b.resid, b.insertion, b.residue)
        bead = (str(b.chain).strip(), int(b.resid), str(b.insertion).strip(), b.name)
        s.add_atom(residue, name=b.name, pos=tuple(float(x) for x in b.position),
                   charge=float(charge.get(bead, 0.0)))  # fmt: skip
    return s


def _unknown_residues(message: str) -> list[str]:
    """The residue names a mapper's error says it has no mapping for."""
    m = re.search(r"(?:protein residues|map has no)[:]?\s+([A-Za-z0-9_, ]+?);", message)
    return [x.strip() for x in m.group(1).split(",") if x.strip()] if m else []


def _renumber_like(cg, system, atoms) -> None:
    """Give the CG residues the chain names and numbers of the all-atom residues.

    Both mappers keep residues in order, one CG residue per all-atom residue,
    so the k-th CG residue is the k-th all-atom residue of ``atoms``.
    """
    ids = system.select(atoms).ids
    res = np.unique(np.asarray(system.atoms["residue"])[ids])
    if len(res) != cg.nresidues:
        return  # the mapper merged or split residues; keep its numbering
    resid = np.asarray(system.residues["resid"])[res]
    ins = np.asarray(system.residues["insertion"])[res]
    chain_names = np.asarray(system.chains["name"])[np.asarray(system.residues["chain"])[res]]
    for r in range(cg.nresidues):
        cg.residues["resid"][r] = int(resid[r])
        cg.residues["insertion"][r] = str(ins[r])
    cg_chain_of = np.asarray(cg.residues["chain"])
    for c in range(cg.nchains):
        first = np.flatnonzero(cg_chain_of == c)
        if len(first):
            cg.chains["name"][c] = str(chain_names[first[0]])
