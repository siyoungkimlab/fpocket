---
name: cg-fpocket
description: Run fpocket or mdpocket on coarse-grained proteins (Martini 2, Martini 3, SIRAH) with the flags and pocket score tuned for them, from ~/fpocket/cg (cgpocket.py). Use this whenever the user wants to find, rank or track binding pockets, cavities or cryptic sites in a Martini or SIRAH structure or CG MD trajectory (.gro/.xtc/.dcd with a GROMACS topology), wants to coarse-grain an all-atom structure and look for pockets in it, asks for "CG pocket detection", "mdpocket on my Martini run", the "tuned fpocket parameters" or "cg presets", wants predicted pockets checked against the ligand of a holo structure (distance to the pocket center, coverage, PPc/MOc), or is about to run plain fpocket on beads -- even if they do not mention presets, since fpocket's all-atom defaults rank pockets on beads almost at random.
---

# fpocket on coarse-grained proteins

fpocket was built for atoms. On beads its defaults find pockets but rank the
real binding site first only 2-12% of the time, so `~/fpocket/cg` provides,
per CG model, detection flags plus a refitted pocket score, tuned the way
fpocket itself was (on its 263-complex training set, judged by the pocket
center lying < 4 Å from the ligand) and checked on held-out apo/holo sets.

Everything goes through `cgpocket.py`, which does the parts that are easy to
get wrong by hand: mapping atoms to beads, telling fpocket each bead's
polarity, and passing the preset.

## Before anything

```bash
conda activate boonza                      # boonza maps structures and reads files
cd ~/fpocket/cg
python cgpocket.py flags --model martini3  # prints the preset, or says it is not ready
```

- The presets live in `~/fpocket/cg/presets.json`; read them from there (or via
  `cgpocket.py flags`), not from memory or from this file -- they get retuned.
  A model whose search is unfinished has `"flags": null` and cgpocket refuses
  it; say so rather than substituting another model's flags.
- Use the fpocket build in `~/fpocket/bin/`. The presets pass
  `--score_coefficients`, which only this build has (stock fpocket rejects it).
  Without that option the build behaves exactly like stock fpocket.
- The all-atom entry (`aa`) is fpocket's own defaults, deliberately unchanged.

## Pick the model

Use the CG model the user's structure or simulation is in -- the presets are
not interchangeable, because bead spacing and polarity differ. For an
all-atom structure with no simulation behind it, any model works; Martini 2
and 3 performed about equally on the held-out sets (see below).

## Single structure

```bash
# all-atom in (.mae .pdb .cif ...): mapped to beads with boonza, then fpocket
python cgpocket.py run protein.mae --model martini3 -o out/

# already coarse-grained: give the topology, which carries Martini bead types
# and SIRAH charges (polarity comes from those)
python cgpocket.py run cg.gro --top topol.top --model martini2 -o out/

# only write the fpocket-ready PDB
python cgpocket.py prep protein.pdb --model sirah -o out/protein_sirah.pdb
```

Only protein is kept (waters, ions, ligands and cofactors removed; residues a
model cannot map, e.g. phosphotyrosine, are left out with a warning). Extra
fpocket flags after `--` override the preset: `... --model martini3 -- -i 20`.

**Chain `LIG` holds probes, not protein.** In this user's structures and
trajectories, `chain LIG` is the probes -- ligand copies placed around the
protein (e.g. by boonza swim) -- and must be left out of any pocket analysis.
Probes are often amino acids themselves, so the `protein` keyword or residue
names would pull them in. cgpocket subtracts chain LIG from every selection,
including a `--selection` you pass, and says how many atoms it dropped; do the
same (`and not chain "LIG"`) in any selection you write yourself, e.g. for
boonza, mdpocket inputs or fitting.

Never hand fpocket a CG PDB that cgpocket did not write: fpocket decides
polarity from the element column (C/S apolar, others polar), and a raw bead
PDB's elements are guesses from bead names (BB -> boron, SC1 -> sulfur), so
every pocket would look apolar and the score would be off.

## Checking pockets against a known ligand

When the user has a holo structure (or asks whether a predicted pocket is
"right"), add it to `run`:

```bash
python cgpocket.py run apo.mae --model martini3 -o out/ \
    --holo holo.mae --holo-ligand "resname LIG"      # --holo-top for a CG holo
# a CG input (e.g. a frame of a simulation): add the all-atom apo
python cgpocket.py run frame.gro --top topol.top --model martini3 -o out/ \
    --apo apo.mae --holo holo.mae --holo-ligand "resname LIG"
```

The holo structure is only the answer key: fpocket runs on the input, and
`boonza.superpose` aligns the holo protein on the all-atom apo (or, without
one, on the structure the pockets were found in, beads or atoms): alpha
carbons paired by sequence, pairs more than 2 Å apart pruned as ChimeraX
does, the whole holo -- ligand included -- moved by the fit. No binding-site
residue list is used or needed (do not use the si.csv residue columns). The
ligand is `--holo-ligand`, by default `resname LIG`; chain-LIG probes in the
holo are kept out of the fit. For every pocket, in fpocket's rank order, it
reports the distance from the pocket center (mean of its alpha-sphere
centers) to the nearest ligand atom and to the ligand centroid, and the two
MOc shares (ligand atoms within 3 Å of the pocket's spheres; spheres within
3 Å of the ligand), with fpocket's own verdicts:

- **PPc**: center < 4 Å from a ligand atom;
- **MOc**: > 50% of ligand atoms covered and > 20% of spheres on the ligand.

It prints the first correct rank and the top 10, and writes every pocket to
`<name>_out/<name>_vs_holo.csv`. Report these two criteria, not residue
overlap. Check the printed superposition (alpha carbons kept, RMSD; a
warning is printed under 20 kept or above 3 Å). The fit follows the rigid
core, so on a protein whose domains moved between apo and holo the ligand
sits relative to the core and can be a few Å off the apo pocket -- enough to
flip a 4 Å verdict.

For a whole set of pairs, `benchmark/run_views.py --data <dir with apo/,
holo/, si.csv> --out <dir> --settings default optimized martini2 martini3
sirah` writes `fpocket_<setting>/<apo>_<holo>/` (view.pml, MAE structures,
pockets_vs_holo.csv) and `summary.csv`/`summary.md` per setting; si.csv
gives only the pairs and EXCLUDE notes.

## Looking at the pockets (all-atom view)

Every pocket in a view (single structure or trajectory consensus pocket) is its own
PyMOL object, `pocket_<rank>`, holding its alpha spheres and its label: click it in
the object panel (or `disable pocket_3`) to hide or show that pocket alone.

Pockets are found on beads but should be looked at on atoms: a bead model in
a viewer is unreadable (no cartoon, odd bonds). So `run` writes
`<name>_out/<name>_view.pml` on a black background: the all-atom apo as a
wheat cartoon, the pocket alpha-sphere centers coloured by rank (1 red,
2 orange, 3 yellow, 4 green, 5 cyan, the rest grey) and labelled with their
rank -- plus, with `--holo`, the holo ligand as blue sticks and the aligned
holo protein in light blue (hidden; toggle it on in the object panel), with
PPc/MOc in the labels.

```bash
cd out/<name>_out && pymol <name>_view.pml
```

The all-atom apo is the input itself when the input is all-atom (same frame,
nothing aligned). For a CG input, give it with `--apo apo.mae`: its alpha
carbons are superposed on the beads by sequence (expect ~1 Å for Martini,
whose BB bead is not on the CA). With `--holo`, the holo is then fitted on
the all-atom apo (CA on CA, tighter than on beads), and the distances are
measured from that placement. Point the user to `_view.pml`, not fpocket's
own `<name>.pml`/`.tcl`, which draw the beads (and whose PyMOL script
mis-colours pocket 1 and errors on a nonexistent last pocket -- an upstream
fpocket bug, harmless).

## Trajectory (mdpocket)

```bash
python cgpocket.py traj --top topol.top --coords cg.gro --traj md.xtc \
    --model martini3 -o out/md --run
# options: --stride 10, --start/--stop, --selection "...", --fit "...", --no-fix
# a solute-only DMS (Martini types / SIRAH charges inside) needs no --top:
#   --coords solvated.dms --traj trajectory.dcd
# speed: ~300-700 beads run at ~60 frames/s, so 1000 frames take ~15-30 s
```

It keeps the protein beads only (no solvent, ions or chain-LIG probes;
mdpocket needs the trajectory to match the topology PDB atom for atom), makes
the protein whole across the box, fits
every frame on the first by the protein backbone beads (BB / GC / CA; never
the probes) -- essential, since mdpocket accumulates its maps on a fixed grid
and raw frames drift tens of Å and rotate freely -- writes
`md.pdb` + `md.dcd`, then runs mdpocket with the preset. Without `--run` it
prints the mdpocket command instead.

With the all-atom structures it also writes a view of the maps (the CG
trajectory itself is never shown):

```bash
python cgpocket.py traj --coords solvated.dms --traj trajectory.dcd --model sirah -o out/mdpocket/md \
    --run --rank --stride 2 --apo apo.mae --holo holo.mae --view-dir out/   # --holo-ligand, default "resname LIG"
```

`--apo` is superposed on the trajectory's reference frame (the fitted first
frame, where mdpocket's maps live) and `--holo` on the apo, both by
`boonza.superpose`. `out/view.pml` shows the apo (wheat), the ligand (blue
sticks), the pocket-frequency map as a red surface at 0.5 (open in at least
half the frames; `isolevel pocket_frequency, 0.3` shows rarer pockets) and the
density as a yellow mesh at the model's calibrated level.
`out/ligand_site_frequency.csv` gives, per ligand atom, the largest pocket
frequency and density within 2 Å: how often the known site is a pocket over the
trajectory.

Structures are written back in the format they were given (MAE stays MAE,
ligand included; CIF stays CIF), except DMS, which is written as MAE so that
PyMOL opens it.

Add `--rank` to `traj` for a pocket ranking over the trajectory: fpocket (with the
preset) runs on every (strided) frame, each pocket's score becomes a probability
(the refitted score is logistic) and gets boonza.sites' burial (share of 26
directions that meet protein within 10 Å). Pockets of all frames are grouped
into consensus pockets (greedy, best first, centres within 4 Å of a running
centroid, `--consensus-cutoff` to merge more or split more; each frame counts
once, with its best member), and ranked three ways:

- **persistence**: mean probability over all frames (0 where absent) -- the
  pockets that are there and good most of the time;
- **quality**: 90th percentile of the probability over the frames it is open,
  pockets open in < 5% of frames ranked after the rest -- good pockets however
  rarely they open (cryptic ones included);
- **quality x buriedness**: quality times mean burial.

A consensus pocket with no fpocket pocket of the apo crystal (mapped to beads)
within 4 Å is flagged **cryptic**. Outputs in the view folder:
`consensus_pockets.csv` (all ranks, occupancy, burial, cryptic, centre),
`pockets_vs_holo.csv` (the same plus distances to the holo ligand, MOc shares,
PPc/MOc and the share of open frames passing PPc), `fpocket_info.txt` (each
consensus pocket's descriptors in its best frame), `frames.csv`, and
`consensus_pockets.pqr` (shown in `view.pml`, coloured by quality rank,
labelled with occupancy and the cryptic flag). Cost: one fpocket run per frame,
~0.1-0.5 s each, so prefer `--stride` for long trajectories.

Read the maps this way:
- `md_freq_iso_0_5.pdb` (fraction of frames with a pocket there): no change
  for CG.
- Density: mdpocket contours it at 8, a level set for all-atom; beads give
  fewer alpha spheres, so cgpocket also writes `md_dens_iso_<level>.pdb` at
  the model's calibrated level (`mdpocket_density_iso` in presets.json, about
  5 for Martini). Use that one, or contour `md_dens.dx` at that level.

For Martini 2 always pass `--top`: its backbone bead types change with
secondary structure, so the built-in type table used without a topology is
only approximate.

## Reading the results

- `<name>_out/<name>_info.txt`: pockets in rank order. `Score` is the CG score
  that did the ranking. Ignore `Druggability Score` on beads: it was trained
  on atoms and is not refitted.
- `<name>_out/<name>_out.pdb` and `pockets/pocketN_atm.pdb` / `_vert.pqr`:
  the pockets for viewing; residue numbers and chains are the input's.
- Look at the top 3-5 pockets, not only the first: on apo structures the
  right one is first in only about a third of cases (table below).

Held-out accuracy (fraction of structures whose first 1 / 3 / 5 / 10 pockets
include one centered < 4 Å from the ligand); `presets.json` has the current
numbers under `validation`:

| | fpocket 48 apo | fpocket 48 holo | Schrödinger 41 apo |
|---|---|---|---|
| all-atom, fpocket defaults | 0.31 / 0.56 / 0.67 / 0.79 | 0.50 / 0.71 / 0.81 / 0.88 | 0.20 / 0.39 / 0.49 / 0.66 |
| Martini 2 preset | 0.60 / 0.81 / 0.88 / 0.94 | 0.73 / 0.90 / 0.98 / 0.98 | 0.37 / 0.49 / 0.56 / 0.63 |
| Martini 3 preset | 0.62 / 0.79 / 0.90 / 0.90 | 0.69 / 0.90 / 0.90 / 0.92 | 0.37 / 0.51 / 0.59 / 0.61 |
| SIRAH preset | 0.65 / 0.77 / 0.81 / 0.88 | 0.62 / 0.83 / 0.85 / 0.90 | 0.41 / 0.51 / 0.56 / 0.68 |

Much of the gap to all-atom defaults is the score, not the resolution:
all-atom with a refitted score reaches 0.56 / 0.77 / 0.92 / 0.96 on the
48 apo structures and 0.27 / 0.51 / 0.63 / 0.73 on the Schrödinger apo set. With ~40-50 structures per column, differences under
~0.1 are within noise.

## Things not to do

- Do not use `-p` above 0: a precedence bug in fpocket's `refine.c` then drops
  every pocket that is not 100% apolar.
- Keep single-linkage clustering (`-C s`, in the presets): the other linkages
  are ~10x slower on beads, which adds up over a trajectory.
- Do not change fpocket's all-atom defaults or built-in score; the CG presets
  are opt-in flags.

## Retuning or extending

The benchmark that produced the presets is in `~/fpocket/cg/benchmark/`
(`prepare.py`, `tune_score.py`, `report_score.py`, `calibrate_density.py`);
`~/fpocket/cg/README.md` explains the method, data sets and caveats. Read it
before retuning, adding a CG model, or answering how the presets were made.
