# fpocket and mdpocket on coarse-grained proteins

fpocket was built and tuned for all-atom structures. On Martini or SIRAH beads
its defaults still find pockets, but rank the real binding site first only
4–12% of the time. This directory adapts fpocket to Martini 2, Martini 3 and
SIRAH proteins:

- it writes beads in a form fpocket can read: polarity in the element column,
  standard residue names;
- it provides, per model, detection flags **and a refitted pocket score**,
  tuned the way fpocket itself was tuned (on its own training set, judged by its
  own criteria);
- it shows the pockets on the all-atom structures, with the holo ligand, as a
  PyMOL session.

fpocket's all-atom behaviour is unchanged. The CG presets reach it through two
opt-in options.

## Use

```bash
conda activate boonza          # boonza maps the structures and reads the files

# all-atom in (.mae .pdb .cif ...): mapped onto beads, fpocket with the preset,
# and an all-atom PyMOL view of the pockets
python cgpocket.py run apo.mae --model sirah -o out/

# the same, checked against a holo structure's ligand (PPc, MOc for every pocket),
# with the view and its structures written to one folder (MAE in, MAE out)
python cgpocket.py run apo.mae --model martini3 -o out/fpocket \
    --holo holo.mae --holo-ligand "resname LIG" --view-dir out/

# a structure that is coarse-grained already: the topology gives Martini bead
# types and SIRAH charges; --apo gives the all-atom structure for the view
python cgpocket.py run frame.gro --top topol.top --model martini3 -o out/ --apo apo.mae

# a CG trajectory: protein beads only, made whole, fitted on the first frame,
# written as PDB + DCD, then mdpocket with the preset
python cgpocket.py traj --top topol.top --coords cg.gro --traj md.xtc \
    --model martini3 -o out/md --run

# just the flags, to pass to fpocket or mdpocket yourself (needs ../bin/fpocket)
python cgpocket.py flags --model sirah
```

Flags after `--` override the preset, e.g. `... run x.mae --model sirah -- -i 25`.

A whole apo/holo set goes through `benchmark/run_views.py` (see
[Batch runs](#batch-runs)). The Claude Code skill `cg-fpocket`
(`skill/cg-fpocket/SKILL.md`; install with
`cp -r skill/cg-fpocket ~/.claude/skills/`) describes the same workflow.

## Presets (`presets.json`)

| model | detection flags | Martini N beads | mdpocket density level |
|---|---|---|---|
| all-atom | fpocket's defaults, built-in score (unchanged) | | 8 |
| Martini 2 | `-m 4.25 -M 7.34 -D 3.21 -i 5 -A 2` | polar | 5 |
| Martini 3 | `-m 4.11 -M 7.14 -D 3.99 -i 10 -A 2` | apolar | 5 |
| SIRAH | `-m 4.00 -M 7.12 -D 2.81 -i 11 -A 2` | | 5.5 |

fpocket's defaults are `-m 3.4 -M 6.2 -D 2.4 -i 15 -A 3`.

**Score.** Each CG preset also passes `--score_coefficients=…`: the pocket
score refitted for that model (see [Method](#method)). That option, and
therefore the presets, need the fpocket build in `../bin`.

**Clustering.** All presets use single-linkage clustering (`-C s -e e`). The
other linkages are about 10× slower on beads, which adds up over a trajectory.

**Source of truth.** `presets.json` holds the flags, the score coefficients,
the training and validation numbers, and the density levels.
`cgpocket.py flags` reads it.

### Why the flags change

An alpha sphere's radius is the distance from its center to the four atom
centers it touches. Beads are farther apart than atoms (nearest-neighbour
median: all-atom 1.37 Å, SIRAH 1.46 Å, Martini 2.8–3.0 Å), so the same
physical cavity gives larger spheres, and fewer of them. All three models end
up with `-m` ≈ 4.0–4.25 and `-M` ≈ 7.1–7.4.

### Bead polarity

fpocket calls an atom apolar when its element's electronegativity is below
2.8 (C, S). Beads are written with element C (apolar) or O (polar):

- **Martini:** by bead type. C and X are apolar; P, Q and D are polar. The N
  class was a tuning choice, and both final presets count it as apolar.
- **SIRAH:** the backbone goes by name (GN and GO polar, GC apolar). A
  side-chain bead is polar when it carries a partial charge (|q| ≥ 0.1), using
  the charges SIRAH itself assigns (boonza's `sirahize`, its residue library,
  or the topology the beads were loaded with). The bead names mislead here:
  Lys BCG/BCE, Arg BCZ, the Asp/Glu carboxylate carbons (BCG/BCD), a free
  Cys BSG and Tyr BCE1/BCE2 (the hydroxyl) are named as carbons or sulfur but
  are charged.

Residue names are written as the standard amino acids, which fpocket's
polarity score expects.

### Probes, hydrogens, missing atoms

- **Probes.** Chain `LIG` holds probes (ligand copies placed around the
  protein, e.g. by boonza swim), not protein. Every selection in `cgpocket.py`
  and `cgprep.py` subtracts it.
- **Hydrogens.** fpocket ignores hydrogens: the Voronoi tessellation, the
  neighbour grid and the surface areas all use heavy atoms only, and on 1FVR
  with or without hydrogens the output is identical. For CG they matter a
  little through the mapping: Martini 3 types histidine ring beads by
  protonation, and SIRAH places a few beads on hydroxyl and indole
  hydrogens.
- **Missing heavy atoms** matter much more. A truncated side chain leaves a
  hole fpocket reports as a pocket, and martinize refuses the residue. Raw
  crystal structures should be completed first; `benchmark/prepare.py --fix`
  uses PDBFixer, heavy atoms only, and adds no loops.

## Looking at the pockets

Pockets are found on beads but should be looked at on atoms. `run` writes
`view.pml` (with `--view-dir`) or `<name>_out/<name>_view.pml`:

- black background, the all-atom apo as a wheat cartoon;
- each pocket's alpha-sphere centers coloured by rank (1 red, 2 orange,
  3 yellow, 4 green, 5 cyan, the rest grey) and labelled with the rank;
- with `--holo`: the ligand as blue sticks, and the aligned holo protein in
  light blue (loaded but hidden; toggle it on), with PPc/MOc in the labels of
  correct pockets.

Each pocket is its own PyMOL object, `pocket_<rank>` (its alpha spheres and its
label), so it can be shown or hidden alone from the object panel.

Structures are written as MAE when the inputs are MAE, so bond orders are
kept. Images exported from PyMOL have a transparent background.

fpocket's own `<name>.pml`/`.tcl` show the beads, so use the view instead.
fpocket's PyMOL template also has an upstream off-by-one bug: pocket 1 is left
uncoloured and the script errors on a nonexistent last pocket. It's harmless
and left unfixed, to keep default fpocket output unchanged.

For trajectories, `traj --run --apo apo.mae [--holo holo.mae] --view-dir out/`
writes the same kind of view for mdpocket's maps:

- `--apo` is superposed on the trajectory's reference frame (the fitted first
  frame), and `--holo` on the apo, both by `boonza.superpose`;
- the pocket-frequency map is a red surface at 0.5 (a pocket in at least half
  the frames);
- the alpha-sphere density is a yellow mesh at the model's calibrated level;
- `ligand_site_frequency.csv` gives, per holo ligand atom, the largest pocket
  frequency and density within 2 Å.

Every frame is fitted on the first by the protein backbone beads (BB / GC /
CA, never chain-LIG probes) before mdpocket runs. Raw frames drift tens of Å
and rotate freely, which would make the maps meaningless. A solute-only DMS
(with Martini types or SIRAH charges in it) needs no topology, and on 1FXX
1000 frames took 10–20 s.

### Pocket ranking over a trajectory (`traj --rank`)

Add `--rank` to `traj` for a pocket ranking over the trajectory: fpocket (with the
preset) runs on every (strided) frame, each pocket's score becomes a probability
(the refitted score is logistic) and gets boonza.sites' burial (share of 26
directions that meet protein within 10 Å). Pockets of all frames are grouped
into consensus pockets (greedy, best first, centres within 4 Å of a running
centroid; each frame counts once, with its best member), and ranked three ways:

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

Structures are written back in the format they were given (MAE stays MAE,
ligand included); DMS is written as MAE, which PyMOL opens.

mdpocket's frequency map (`*_freq_iso_0_5.pdb`) needs no change for CG. Its density map is contoured at 8, a level calibrated for all-atom;
`traj --run` also writes the contour at the model's calibrated level
(`<prefix>_dens_iso_<level>.pdb`).

## Checking pockets against a holo ligand

With `--holo`, the holo structure is only the answer key. `boonza.superpose`
aligns the holo protein on the all-atom apo (or, without one, on the structure
the pockets were found in), and the ligand moves with it:

- alpha carbons are paired by sequence;
- pairs that stay more than 2 Å apart are pruned, as ChimeraX does;
- no binding-site residue list is used.

For every pocket, in fpocket's rank order, `pockets_vs_holo.csv` gives:

- the distance from the pocket center (the mean of its alpha-sphere centers)
  to the nearest ligand heavy atom and to the ligand centroid;
- the share of ligand atoms within 3 Å of the pocket's spheres;
- the share of spheres within 3 Å of the ligand;
- fpocket's own verdicts:
  - **PPc:** center < 4 Å from a ligand atom;
  - **MOc:** > 50% of the ligand covered and > 20% of the spheres on the
    ligand.

The fit follows the rigid core. On a protein whose domains moved between apo
and holo, the ligand sits relative to the core and can be a few Å off the apo
pocket.

## Method

The presets were made the way fpocket's own flags and score were made. The
data and criteria come from `fpocket-data-1.0`
(sourceforge.net/projects/fpocket); everything is in `benchmark/`.

1. **Training set:** fpocket's 263 holo complexes (`data/train`, with
   `train-t.txt`), completed with PDBFixer (heavy atoms only, every residue
   PDBFixer dropped copied back). Each complex was mapped to Martini 2,
   Martini 3 and SIRAH. The site is the ligand itself.
2. **Criterion:** PPc, the pocket center within 4 Å of a ligand atom, as
   tpocket applies it. MOc is reported alongside.
3. **Score refit:** fpocket ranks pockets by a linear score of pocket
   descriptors (its current score is a logistic regression).
   `--write_score_descriptors` makes fpocket write each pocket's descriptors:
   the union of those in the current and earlier PLS scores (`nas_norm`,
   `prop_asapol_norm`, `mean_loc_hyd_dens_norm`, `polarity_score`,
   `as_density`, `convex_hull_volume`, `surf_pol_vdw14`, `surf_apol_vdw14`).
   A class-balanced logistic regression on standardized descriptors, labelled
   by PPc, is converted back to raw-descriptor coefficients for
   `--score_coefficients`.
4. **Search** (`tune_score.py`): 150 random and 100 local trials per model
   over `-m -M -D -i -A` (and the Martini N class). Each trial runs fpocket on
   all 263 complexes and refits the score. The objective is the mean of Top-1
   and Top-3 with the score fitted on the other folds (5-fold, complexes held
   out whole), the paper's T1/T3.
5. **Choice** (`report_score.py`): with 263 complexes the single best trial
   is partly luck, so each preset is the trial whose 8 nearest neighbours in
   parameter space score best. Its score is then refitted on all 263.
6. **Validation** on sets the search never saw:
   - fpocket's 48 apo/holo pairs (`pp-apo-t.txt`), with the dataset's own
     apo/holo superposition;
   - the Schrödinger set: the 41 pairs of `si.csv` not marked EXCLUDE, with
     the holo `LIG` placed by `boonza.superpose` of the whole proteins. si.csv
     is read only for the pairs.

**Pipeline check.** With fpocket's default flags and the original PLS score
(the coefficients left in `src/pscoring.c`), the training set gives T1/T3 =
0.64 / 0.85 under PPc. The paper reports 62/86, so the data and criteria
reproduce fpocket's published training numbers. fpocket 4's current built-in
score, derived from druggability, gives 0.41 / 0.61 on the same set.

## Results

Each cell is Top-1 / Top-3 / Top-5 / Top-10: the fraction of structures with a
correct pocket among the first k pockets. With 40–50 structures per column,
differences under about 0.07–0.1 are within noise.

**Training set** (263 holo), score cross-validated, PPc:

| model | T1 / T3 / T5 / T10 |
|---|---|
| all-atom, fpocket defaults | 0.41 / 0.61 / 0.70 / 0.83 |
| all-atom, default flags + refitted score | 0.65 / 0.85 / 0.91 / 0.93 |
| Martini 2 | 0.63 / 0.83 / 0.87 / 0.92 |
| Martini 3 | 0.68 / 0.84 / 0.87 / 0.89 |
| SIRAH | 0.60 / 0.79 / 0.84 / 0.88 |

The CG rows also had their flags chosen on this set, so they are slightly
optimistic.

**Held out, PPc:**

| model | fpocket 48 apo | fpocket 48 holo | Schrödinger 41 apo |
|---|---|---|---|
| all-atom, fpocket defaults | 0.31 / 0.56 / 0.67 / 0.79 | 0.50 / 0.71 / 0.81 / 0.88 | 0.20 / 0.39 / 0.49 / 0.66 |
| all-atom, refitted score | 0.56 / 0.77 / 0.92 / 0.96 | 0.73 / 0.92 / 0.96 / 0.96 | 0.27 / 0.51 / 0.63 / 0.73 |
| Martini 2, fpocket defaults | 0.10 / 0.15 / 0.25 / 0.42 | 0.08 / 0.19 / 0.25 / 0.35 | 0.07 / 0.15 / 0.27 / 0.41 |
| Martini 3, fpocket defaults | 0.06 / 0.19 / 0.23 / 0.44 | 0.04 / 0.25 / 0.33 / 0.46 | 0.10 / 0.20 / 0.20 / 0.41 |
| SIRAH, fpocket defaults | 0.10 / 0.23 / 0.33 / 0.52 | 0.12 / 0.21 / 0.38 / 0.54 | 0.10 / 0.17 / 0.27 / 0.41 |
| **Martini 2 preset** | 0.60 / 0.81 / 0.88 / 0.94 | 0.73 / 0.90 / 0.98 / 0.98 | 0.37 / 0.49 / 0.56 / 0.63 |
| **Martini 3 preset** | 0.62 / 0.79 / 0.90 / 0.90 | 0.69 / 0.90 / 0.90 / 0.92 | 0.37 / 0.51 / 0.59 / 0.61 |
| **SIRAH preset** | 0.65 / 0.77 / 0.81 / 0.88 | 0.62 / 0.83 / 0.85 / 0.90 | 0.41 / 0.51 / 0.56 / 0.68 |

**Held out, MOc:**

| model | fpocket 48 apo | fpocket 48 holo | Schrödinger 41 apo |
|---|---|---|---|
| all-atom, fpocket defaults | 0.25 / 0.46 / 0.56 / 0.65 | 0.48 / 0.67 / 0.75 / 0.83 | 0.07 / 0.22 / 0.27 / 0.32 |
| all-atom, refitted score | 0.46 / 0.65 / 0.79 / 0.83 | 0.69 / 0.90 / 0.94 / 0.96 | 0.17 / 0.27 / 0.32 / 0.37 |
| **Martini 2 preset** | 0.58 / 0.77 / 0.79 / 0.81 | 0.69 / 0.77 / 0.85 / 0.85 | 0.24 / 0.32 / 0.37 / 0.37 |
| **Martini 3 preset** | 0.62 / 0.79 / 0.88 / 0.88 | 0.73 / 0.92 / 0.92 / 0.94 | 0.32 / 0.41 / 0.41 / 0.46 |
| **SIRAH preset** | 0.56 / 0.67 / 0.71 / 0.75 | 0.60 / 0.83 / 0.88 / 0.90 | 0.27 / 0.32 / 0.34 / 0.37 |

Reading the tables:

- **The refitted score does most of the work.** With fpocket's defaults, CG
  proteins rank the site first in 4–12% of structures.
- **Fair comparison.** The fair all-atom reference is the refitted-score row.
  Against it, the three CG presets are about as good on the 48-protein set.
  On the Schrödinger apo set they are ahead at Top-1 (0.32–0.41 vs 0.27),
  level at Top-3 (0.51), and behind by Top-10 (0.61–0.68 vs 0.73).
- **Apo is harder for every representation.** Many of the Schrödinger apo
  sites are partly closed.

## Batch runs

```bash
python benchmark/run_views.py --data ~/Dropbox/PocketFinding/SchrodingerSet \
    --out ~/Dropbox/PocketFinding/SchrodingerSet/fpocket \
    --settings default optimized martini2 martini3 sirah
```

For each pair of `si.csv` (not marked EXCLUDE) and each setting, this writes
`fpocket_<setting>/<apo>_<holo>/`:

- `view.pml`;
- `apo.mae`, `holo.mae`, `ligand.mae`, `pockets.pqr`;
- `pockets_vs_holo.csv`, `fpocket_info.txt`;
- fpocket's own run in `fpocket/`.

Each setting also gets `summary.csv` (first correct rank per pair) and
`summary.md` (Top-1/3/5/10).

The settings are:

- `default`: all-atom fpocket as shipped;
- `optimized`: all-atom with fpocket's flags and the refitted score
  (`benchmark/results/aa_refit_reference.json`); all-atom detection flags were
  not searched;
- `martini2`, `martini3`, `sirah`: the presets.

## Reproducing

```bash
cd benchmark
# data: fpocket-data-1.0 (train/, pp_data/, train-t.txt, pp-apo-t.txt) into data/train263 and data/pp48
PYTHONPATH=/path/to/pdbfixer python prepare.py train263 --data data/train263 --fix
PYTHONPATH=/path/to/pdbfixer python prepare.py pp48 --data data/pp48 --fix
python prepare.py schrodinger --data ~/Dropbox/PocketFinding/SchrodingerSet
for m in martini2 martini3 sirah; do python tune_score.py --model $m --random 150 --local 100 -j 4; done
python report_score.py --models martini2 martini3 sirah   # presets + validation -> ../presets.json
python calibrate_density.py --set pp48                    # mdpocket density levels -> ../presets.json
```

Trials go to `runs/train263/<model>.jsonl`, with each trial's pocket
descriptors in `runs/train263/<model>/*.npz` for refitting without rerunning
fpocket. `tune.py` and `report.py` are from earlier searches (on the
48-protein set, a residue-overlap criterion); `tune_score.py` and
`report_score.py` reuse their helpers.

## Changes to fpocket itself

- **`--score_coefficients=c0,…,c8`** and **`--write_score_descriptors`**
  (long-only, opt-in). Without them fpocket's output is unchanged: on all 39
  sample structures every output file is identical to the original binary's
  apart from the Monte Carlo volume lines, which vary between runs of the
  original too. The built-in coefficients passed through the option reproduce
  the default output exactly.
- **`src/mdparams.c`:** mdpocket copied each command-line argument into an
  8-byte buffer and so crashed on most command lines (heap overflow). Fixed.
- **Not changed:** `src/refine.c` compares the apolar ratio with a boolean
  (`pasph < (p || as_density < min)`), so `-p` above 0 drops every pocket that
  isn't 100% apolar. The presets keep `-p 0`. A single-sphere pocket
  (`-i 1`) gets a NaN density and score; the searches keep `-i ≥ 2`.
- **Tests:** 8 of the repository's 11 tests fail identically with the original
  binary, so these failures predate this work.

## Caveats

- **SIRAH beads** come from boonza's `sirah.map_structure`, not from
  `sirahize`. `sirahize` keys beads by residue number without the insertion
  code and fails on proteins numbered like `77`/`77A` (serine proteases and
  others). The bead positions are the same. Charges come from `sirahize` or,
  where it fails (1IGJ), from SIRAH's residue library.
- **Unmappable residues.** Residues a model cannot map (phosphotyrosine,
  D-amino acids, modified residues) are left out with a warning.
- **Martini 2 types.** Backbone bead types change with secondary structure, so
  give a topology (`--top`); the built-in type table (`bead_types.json`) is
  approximate without one. The side chains of Leu and Ile (AC1) and Val (AC2)
  use Martini 2's amino-acid prefix; before 2026-10-06 they were wrongly
  counted polar. The Martini 2 preset was re-searched after the fix (a local
  search around the old preset; the old runs are in
  `benchmark/runs/*/archive_martini2_ac_polar/`).
- **Surface descriptors.** The score's surface descriptors use element radii,
  which don't fit beads; the refitted coefficients absorb part of that.
- **Tests:** `tests/` needs numpy and pytest; two tests need boonza.
