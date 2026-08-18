# finetuned-mlips

Finetuning the [mace-mh-1](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1) foundation model on Ag/C/H/O/Ni surface catalysis DFT data related to ethylene epoxidation (ethylene oxide (EO) reactivity and O₂ dissociation on Ag(-Ni) single-atom alloy (SAA) surfaces). The result of this project pipeline is a MLIP specialized for this chemistry while retaining the foundation model's accuracy on its original training distribution, via multihead replay finetuning.

This project supports two finetuning approaches: direct finetuning, and Optuna-driven hyperparameter search evaluated via 4-fold cross-validation (CV). The former is computationally cheaper but the latter is recommended to reduce variance from a single train/valid split.

See `training-data/README.md` for information on each dataset file.

See the MACE fine-tuning paper that informed most of the decisions made in this project [here](https://arxiv.org/abs/2605.09394).

## Project Pipeline

| # | Location | Stage |
|---|----------|-------|
| 1 | `data-processing/` | Pull raw VASP relaxations from Box/DropBox, filter to relaxations on select elements |
| 2 | `mace-mh-1/build-training-set.ipynb` | Write selected OUTCARs to a `.xyz` to build the multihead replay training set |
| 3 | `mace-mh-1/cv-optuna/cv-setup.ipynb` | Set up K-fold cross-validation + Optuna hyperparameter search (see `mace-mh-1/loni-setup.md`) |
| 4 | `mace-mh-1/cv-optuna/optuna_post-processing.ipynb` | Final training run on the full dataset with the best hyperparameters |
| 5 | Same as above, `test_eval.py` | Evaluate the finetuned model on the held-out test set, return to post-processing notebook to visualize eval metrics |

## Environment setup

### Python environment

```bash
conda env create -f mace-mh-1/mace-mh-1.yml
conda activate mace-mh-1
```

See `mace-mh-1/loni-setup.md` to set up the environment on LONI.

### Misc. Training Notes

**Head naming vs. log output:**

`HEAD_NAME = "tuned_head"` (and `--head_ft tuned_head` passed to
`fine_tuning_select`) is only the label used within this notebook/pipeline. Regardless of what you name it, `run_train`'s per-epoch log output always reports it as head `Default`, and the replay head as `pt_head` — these names are fixed by MACE, not configurable. (`parse_log()` in `cv_utils.py` expects `head="Default"` for this reason.)

**mace.cli.run_train flags** 

`--train-file` --> the target/finetuning dataset you created (becomes Default head) \
`--pt-train-file` --> the replay/pretraining dataset, created by selecting from the .xyz of replay configs (becomes pt_head) 
