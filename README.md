# finetuned-mlips

Finetuning the [mace-mh-1](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1) foundation model on Ag/C/H/O/Ni surface catalysis DFT data related to ethylene epoxidation (ethylene oxide (EO) reactivity and O₂ dissociation on Ag(-Ni) single-atom alloy (SAA) surfaces). The result of this project pipeline is a MLIP specialized for this chemistry while retaining the foundation model's accuracy on its original training distribution, via multihead replay finetuning.

This project supports two finetuning approaches: direct finetuning, and Optuna-driven hyperparameter search evaluated via 4-fold cross-validation (CV). The former is computationally cheaper but the latter is recommended to reduce variance from a single train/valid split.

## Project Pipeline

| # | Location | Stage |
|---|----------|-------|
| 1 | `data-processing/` | Pull raw VASP relaxations from Box/DropBox, filter to relaxations on select elements |
| 2 | `mace-mh-1/build-training-set.ipynb` | Write selected OUTCARs to a `.xyz` to build the multihead replay training set |
| 3 | `mace-mh-1/cv-optuna/` | K-fold cross-validation + Optuna hyperparameter search (see [README](mace-mh-1/cv-optuna/README.md)) |
| 4 | `mace-mh-1/cv-optuns/optuna_post-processing.ipynb` | Final training run on the full dataset with the best hyperparameters, model eval |
| 5 | `mace-mh-1/test_eval.py`, `mace-mh-1/results/post-processing.ipynb` | Evaluate the finetuned model on the held-out test set |

See [`training-data/README.md`](training-data/README.md) for information on each dataset file.

## Environment setup

### Python environment

```bash
conda env create -f mace-mh-1/mace-mh-1.yml
conda activate mace-mh-1
```

Note: Training itself (`mace.cli.run_train`) is run on LONI.

## Repository layout

```
data-processing/       Raw VASP data -> cleaned/screened OUTCARs (see below)
  ag-data-cleaning.ipynb    filter folder names by element/calc type, screen for relaxations
  ag-data-selection.ipynb   download OUTCAR/INCAR, screen for convergence
  cache/                    intermediate folder-name lists, per source dataset
  incars/, outcars/         downloaded VASP files (gitignored)

training-data/          All .xyz datasets used for finetuning/CV/eval (see training-data/README.md)

mace-mh-1/               Finetuning pipeline for the mace-mh-1 foundation model
  build-training-set.ipynb   OUTCAR -> .xyz, build replay/finetuning dataset, launch training
  cv-optuna/                 cross-validation + hyperparameter search (see cv-optuna/README.md)
  full_train.sh              final training run (SLURM)
  test_eval.py, test_eval.sh single-point eval of a trained model vs. DFT reference
  results/                    eval outputs + post-processing notebook
  models/                     trained model checkpoints (gitignored) + model card
```

---

## 1. Data processing (`data-processing/`)

**Goal:** convert raw VASP relaxation runs stored in Box/DropBox into a clean set of INCAR/OUTCAR pairs for stage 2 to convert into training data.

DFT runs originate from two source datasets (a Box folder and a DropBox folder), each containing hundreds of subfolders — one per DFT calculation — named e.g. `Ag100_OMC` or `EO_subO`. Most subfolders are not relaxations suitable for training (wrong elements, single-point calculations, dimer/NEB searches, exploratory or known-bad runs), so most of this stage is filtering before any downloading occurs.

Notebooks run in order:

1. **`ag-data-cleaning.ipynb`** — lists all subfolders in the source, then filters in passes:
   - drops folders flagged by name as bad/excluded runs (e.g. `wrong`, `exploded`) or as unwanted elements/calc types (e.g. other metals, bulk/gas-phase references, bare slabs)
   - opens each remaining folder's `POSCAR`, keeps only runs whose elements are a subset of `{Ag, C, H, O, Ni}`
   - opens each folder's `INCAR`, keeps only genuine ionic relaxations — the primary filter, since folder names alone are unreliable. A run qualifies as a relaxation if it uses a relaxation algorithm (`IBRION`), is not spin-polarized, exceeds a handful of ionic steps, and is not tagged as a dimer/NEB-type calculation.

   Output at each stage is a plain text list of surviving folder names, cached under `cache/<source_dataset>/*_foldernames.txt`, allowing inspection or reruns from any stage without re-querying the Box/DropBox API.

2. **`ag-data-selection.ipynb`** — for folders surviving cleaning, downloads `INCAR`/`OUTCAR` files (into `incars/` and `outcars/`), then applies a second, numerical screening pass:
   - drops runs that hit VASP's ionic-step limit (`NSW`) without converging, or that stopped after only a handful of steps
   - plots each run's energy trajectory for visual convergence check, with manual removal of runs that still appear incorrect

   The final list of accepted folder names is cached as `screened_foldernames.txt` — this is what stage 2 (`build-training-set.ipynb`) reads to determine which downloaded OUTCARs to use.

**Notes:**
- Both notebooks are designed to be re-run per source dataset via a `CACHE`/`PREFIX` variable near the top of each (see notebooks' own markdown for details).
- Box/DropBox authentication cells need to run only once per token refresh, not on every execution — see the "one-time" markdown notes in `ag-data-cleaning.ipynb`.
- This stage is inherently manual/iterative, particularly the final energy-trajectory spot-check. Cells are expected to run out of order with exclude-lists adjusted during inspection, rather than top-to-bottom execution in one pass.