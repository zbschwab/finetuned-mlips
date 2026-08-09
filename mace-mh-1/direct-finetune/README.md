# Direct Finetune (MACE-MH-1 OMAT)

Single `run_train` job on all training data — also useful to get a sense of
where to set Optuna's parameter search range. Use `cv-optuna/` instead for the
tuned/validated path.

See all available parameters and default paths with:
```bash
conda activate mace-mh-1
python -m mace.cli.run_train --help
```

See the [docs](https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html) or [troubleshooting](https://mace-docs.readthedocs.io/en/latest/guide/troubleshooting.html).

## Before running
- Fill in parameters at the top of the script (`ENERGY_WEIGHT` through `SWA_LR`).
- Confirm `train_withNi.xyz` / `selected_configs_withNi.xyz` exist under
  `/work/<user>/mace-finetune/data` on the machine you plan to train on.

## Run on LONI
```bash
scp direct_tune.sh <user>@qbc1:/home/<user>/mace-finetune/scripts/
cd /home/<user>/mace-finetune/scripts/
sbatch direct_tune.sh
```