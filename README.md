# finetuned-mlips

Finetuning the [mace-mh-1](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1) foundation model on Ag/C/H/O/Ni surface catalysis DFT data related to ethylene epoxidation (ethylene oxide (EO) reactivity and O₂ dissociation on Ag(-Ni) single-atom alloy (SAA) surfaces). The result of this project pipeline is a MLIP specialized for this chemistry while retaining the foundation model's accuracy on its original training distribution, via multihead replay finetuning.

This project supports two finetuning approaches: direct finetuning, and Optuna-driven hyperparameter search evaluated via 4-fold cross-validation (CV). The former is computationally cheaper but the latter is recommended to reduce variance from a single train/valid split.

## Project Pipeline

| # | Location | Stage |
|---|----------|-------|
| 1 | `data-processing/` | Pull raw VASP relaxations from Box/DropBox, filter to relaxations on select elements |
| 2 | `mace-mh-1/mace-ft-pipeline.ipynb` | Write selected OUTCARs to a `.xyz` to build the multihead replay training set |
| 3 | `mace-mh-1/cv-optuna/` | K-fold cross-validation + Optuna hyperparameter search (see [README](mace-mh-1/cv-optuna/README.md)) |
| 4 | `mace-mh-1/cv-optuns/optuna_post-processing.ipynb` | Final training run on the full dataset with the best hyperparameters, model eval |
| 5 | `mace-mh-1/test_eval.py`, `mace-mh-1/results/post-processing.ipynb` | Evaluate the finetuned model on the held-out test set |

Stages 2, 4, and 5 are documented inline in this README and in their notebooks; stages 1 and 3 get the fuller treatment below since they have the most moving parts. See also [`training-data/README.md`](training-data/README.md) for what each dataset file in this pipeline actually is.

*(Sections for stages 2, 4, and 5 are still being filled in — more detail coming.)*

## Environment setup

### Python environment

```bash
conda env create -f mace-mh-1/mace-mh-1.yml
conda activate mace-mh-1
```

This covers everything needed to run the notebooks and scripts in this repo locally (data pulls, dataset construction, and local MACE inference/eval). Training itself (`mace.cli.run_train`) is run on LONI (Tulane's HPC cluster) using a separate environment set up the same way — see the `module load` / `conda activate` lines at the top of `fold_train.sh` and `full_train.sh` for the cluster-side setup.

### Data-source credentials

`data-processing/` needs a `.env` file (gitignored, never commit this) with Box and DropBox API credentials:

```
DROPBOX_REFRESH_TOKEN=
APP_KEY=
APP_SECRET=
DROPBOX_TOKEN=
BOX_CLIENT_ID=
BOX_CLIENT_SECRET=
DEV_TOKEN=
```

<!-- TODO(zschwab): document how a new group member actually gets these -->

## Repository layout

```
data-processing/       Raw VASP data -> cleaned/screened OUTCARs (see below)
  ag-data-cleaning.ipynb    filter folder names by element/calc type, screen for relaxations
  ag-data-selection.ipynb   download OUTCAR/INCAR, screen for convergence
  cache/                    intermediate folder-name lists, per source dataset
  incars/, outcars/         downloaded VASP files (gitignored)

training-data/          All .xyz datasets used for finetuning/CV/eval (see training-data/README.md)

mace-mh-1/               Finetuning pipeline for the mace-mh-1 foundation model
  mace-ft-pipeline.ipynb    OUTCAR -> .xyz, build replay/finetuning dataset, launch training
  cv-optuna/                 cross-validation + hyperparameter search (see cv-optuna/README.md)
  full_train.sh              final training run (SLURM)
  test_eval.py, test_eval.sh single-point eval of a trained model vs. DFT reference
  results/                    eval outputs + post-processing notebook
  models/                     trained model checkpoints (gitignored) + model card
```

---

## 1. Data processing (`data-processing/`)

**Goal:** turn raw VASP relaxation runs sitting in Box/DropBox into a clean set of INCAR/OUTCAR pairs that stage 2 converts into training data.

The DFT runs come from two source datasets (a Box folder and a DropBox folder), each containing hundreds of subfolders — one per DFT calculation — with names like `Ag100_OMC` or `EO_subO`. Most of these folders are *not* relaxations we want to train on (wrong elements, single-point calculations, dimer/NEB searches, exploratory or known-bad runs), so most of the work here is filtering that pool down before spending time downloading anything.

Run the two notebooks in order:

1. **`ag-data-cleaning.ipynb`** — lists all subfolders in the source, then filters down in a few passes:
   - drop folders whose names flag them as bad/excluded runs (e.g. `wrong`, `exploded`) or as elements/calc types we don't want (e.g. other metals, bulk/gas-phase references, bare slabs)
   - open each remaining folder's `POSCAR` and keep only runs whose elements are a subset of `{Ag, C, H, O, Ni}`
   - open each folder's `INCAR` and keep only genuine ionic relaxations — this is the important filter, since folder names alone aren't reliable. A run counts as a relaxation if it uses a relaxation algorithm (`IBRION`), isn't spin-polarized, takes more than a handful of ionic steps, and isn't tagged as a dimer/NEB-type calculation.

   The output at each stage is a plain text list of surviving folder names, cached under `cache/<source_dataset>/*_foldernames.txt`, so you can inspect or rerun from any stage without re-hitting the Box/DropBox API.

2. **`ag-data-selection.ipynb`** — for the folders that survived cleaning, downloads the actual `INCAR`/`OUTCAR` files (into `incars/` and `outcars/`), then does a second, numerical screening pass:
   - drops runs that hit VASP's ionic-step limit (`NSW`) without converging, or that stopped after only a handful of steps
   - plots each run's energy trajectory so you can eyeball convergence and manually drop anything that still looks wrong

   The final list of accepted folder names is cached as `screened_foldernames.txt` — this is what stage 2 (`mace-ft-pipeline.ipynb`) reads to know which downloaded OUTCARs to actually use.

**A few things worth knowing before you run these:**
- Both notebooks are written to be re-run per source dataset — there's a `CACHE`/`PREFIX` variable near the top of each you set before running (see the notebooks' own markdown for details).
- The Box/DropBox authentication cells only need to be run once per token refresh, not every time — see the "one-time" markdown notes in `ag-data-cleaning.ipynb`.
- This stage is inherently manual/iterative (especially the final energy-trajectory spot-check) — expect to run cells out of order and adjust exclude-lists as you inspect the data, not to execute the notebook top-to-bottom in one pass.
