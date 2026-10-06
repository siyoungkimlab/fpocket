"""Tests of the CG preparation for fpocket: bead polarity, names and PDB columns.

    conda activate boonza && pytest cg/tests
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cgpocket  # noqa: E402
import cgprep  # noqa: E402


@pytest.mark.parametrize(
    ("bead_type", "polar"),
    [("C1", False), ("SC4", False), ("TC5", False), ("X2", False), ("C6v", False),
     ("P1", True), ("SP2", True), ("TP1dq", True), ("Q5", True), ("SQ4p", True), ("Qda", True),
     ("D", True), ("N0", False), ("Nda", False), ("TN6d", False), ("AC1", False), ("AC2", False)],
)  # fmt: skip
def test_martini_polarity(bead_type, polar):
    assert cgprep.martini_polar(bead_type) is polar


def test_martini_n_class_is_a_choice():
    assert cgprep.martini_polar("TN6d", n_polar=True)
    assert not cgprep.martini_polar("TN6d", n_polar=False)


@pytest.mark.parametrize(
    ("name", "charge", "polar"),
    [("GN", 0.13, True), ("GO", -0.23, True), ("GC", 0.10, False), ("GN", 0.4, True),
     ("BCG", 0.0, False), ("BSD", 0.0, False), ("BCE1", 0.0, False),
     ("BCE", 0.6, True), ("BCG", 0.4, True), ("BCD", -0.3, True), ("BSG", -0.2, True),
     ("BCE2", -0.1, True), ("BNE", 0.1, True), ("BOG", -0.4, True)],
)  # fmt: skip
def test_sirah_polarity(name, charge, polar):
    """Backbone by name; a side-chain bead is polar when it is charged."""
    assert cgprep.sirah_polar(name, charge) is polar


def test_sirah_library_charges():
    pytest.importorskip("boonza")
    q = cgprep.sirah_charges(["BCG", "BCE", "BCG", "BCZ", "BSD", "BCE2"],
                             ["sK", "sK", "sL", "sR", "sM", "sY"])  # fmt: skip
    assert q == pytest.approx([0.4, 0.6, 0.0, 0.3, 0.0, -0.1])


@pytest.mark.parametrize(
    ("resname", "standard"),
    [("sA", "ALA"), ("sHe", "HIS"), ("sX", "CYS"), ("HSD", "HIS"), ("CYX", "CYS"), ("GLY", "GLY")],
)
def test_standard_resnames(resname, standard):
    assert cgprep.standard_resname(resname) == standard


def test_guess_model():
    assert cgprep.guess_model(["GN", "GC", "GO"], ["sA"]) == "sirah"
    assert cgprep.guess_model(["BB", "SC1"], ["ALA"]) == "martini"
    assert cgprep.guess_model(["N", "CA", "C", "O"], ["ALA"]) == "aa"


def test_pdb_columns(tmp_path):
    path = tmp_path / "x.pdb"
    cgprep.write_pdb(path, ["BB", "SC1"], ["LEU", "LEU"], ["A", "A"], [1001, 1001], ["", ""],
                     [(1.0, -2.0, 3.0), (10.5, 20.25, -30.125)], ["O", "C"])  # fmt: skip
    lines = [x for x in path.read_text().splitlines() if x.startswith("ATOM")]
    assert lines[0][12:16] == " BB "
    assert lines[1][17:20] == "LEU" and lines[1][21] == "A" and int(lines[1][22:26]) == 1001
    assert float(lines[1][30:38]) == 10.5 and float(lines[1][46:54]) == -30.125
    assert lines[0][76:78].strip() == "O" and lines[1][76:78].strip() == "C"


def test_merge_flags_overrides_the_preset():
    base = ["-m", "4.9", "-M", "7.5", "-i", "9"]
    assert cgpocket.merge_flags(base, ["-i", "20"]) == ["-m", "4.9", "-M", "7.5", "-i", "20"]
    assert cgpocket.merge_flags(base, []) == base


def test_unknown_residues_from_mapper_messages():
    msg = "not martini3001 protein residues: DAS, MSE; leave them out of atoms='protein'"
    assert cgprep._unknown_residues(msg) == ["DAS", "MSE"]
    assert cgprep._unknown_residues("SIRAH's map has no LIG; leave them out") == ["LIG"]


def test_probes_in_chain_lig_are_not_protein():
    """Chain LIG holds probes (ligand copies); they never enter the pocket search."""
    boonza = pytest.importorskip("boonza")
    s = boonza.System("probe")
    for chain, resname in (("A", "ALA"), ("LIG", "ALA")):
        r = s.add_residue(s.add_chain(name=chain), name=resname, resid=1)
        for k, name in enumerate(("N", "CA", "C", "O", "CB")):
            r.add_atom(name=name, anum=7 if name == "N" else 8 if name == "O" else 6, pos=(k, 0.0, 0.0))
    assert cgprep.probe_ids(s).tolist() == [5, 6, 7, 8, 9]
    assert set(cgpocket.protein_ids(s, "all").tolist()) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize(
    ("path", "fmt"),
    [("apo.mae", "mae"), ("apo.maegz", "mae"), ("apo.mae.gz", "mae"), ("apo.dms", "mae"),
     ("apo.cif", "cif"), ("apo.pdb", "pdb"), ("apo.gro", "gro"), ("apo.xyz", "pdb")],
)  # fmt: skip
def test_structures_keep_their_format(path, fmt):
    """The view writes each structure in the format it was given; DMS as MAE."""
    pytest.importorskip("boonza")
    assert cgpocket.structure_format(path) == fmt
