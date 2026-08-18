# LONI setup

Create these directories before your first run:

```bash
mkdir -p /home/<user>/mace-finetune/scripts
mkdir -p /home/<user>/mace-finetune/cv-optuna
mkdir -p /work/<user>/mace-finetune/data/folds
mkdir -p /work/<user>/mace-finetune/logs
```

## Conda environment

`scp` `mace-mh-1.yml` to `/home/<user>/mace-finetune/`.

Set up your LONI-side conda env once, from `/home/<user>/`:

```bash
module load conda
conda env create -f /home/<user>/mace-finetune/mace-mh-1.yml -p /work/<user>/.conda/envs/mace-mh-1
conda activate /work/<user>/.conda/envs/mace-mh-1
python -c "import mace; import ase; print(mace.__version__)"
```

Note: If `conda activate` fails right after creation, your shell likely needs a one-time init (`conda init bash`) followed by restarting your shell or `source ~/.bashrc`.

## tmux (CV/Optuna path only)

`optuna_driver.py` runs for the full duration of the search (multiple hours), polling SLURM in a loop. It must run in a `tmux` session on the login node.

Direct finetuning does not need tmux: `full_train.sh` submits one `sbatch` job and returns immediately.

```bash
tmux new -s optuna
module load conda/23.11.0
source /usr/local/packages/conda/23.11.0/etc/profile.d/conda.sh
conda activate "${MACE_ENV:-/work/$USER/.conda/envs/mace-mh-1}"
cd /home/<user>/mace-finetune/cv-optuna
python -u optuna_driver.py 2>&1 | tee driver.log
# detach: Ctrl-b d
# reattach: tmux attach -t optuna
```

Monitor job status:

```bash
squeue -u <user>
sacct -X -u <user>   # -X restricts to main job allocation, excludes .batch/.extern steps
```

## Running the CV/Optuna search

### 1. Upload inputs
Run from the repo root on your local machine, after `build-training-set.ipynb`
has written `training-data/selected_configs.xyz`
and `cv-setup.ipynb` has written the fold files under `training-data/folds/`:

```bash
scp mace-mh-1/cv-optuna/optuna_driver.py mace-mh-1/cv-optuna/cv_utils.py <user>@qbc1:/home/<user>/mace-finetune/cv-optuna/

scp mace-mh-1/cv-optuna/fold_train.sh <user>@qbc1:/home/<user>/mace-finetune/scripts/

scp training-data/folds/fold*_train.xyz training-data/folds/fold*_valid.xyz <user>@qbc1:/work/<user>/mace-finetune/data/folds/

scp training-data/selected_configs.xyz <user>@qbc1:/work/<user>/mace-finetune/data/
```

### 2. Launch the study
In the `tmux` session set up above, run `optuna_driver.py` (as shown). It
samples one hyperparameter set per trial, submits one SLURM job per fold via
`fold_train.sh`, polls `sacct` until each finishes, and caches completed folds
in `fold_cache.json` and trial results in `optuna_study.log`.
Safely exit session with ctrl+b, then d

### 3. Resume after an interruption
A dropped `tmux` session or killed driver is safe to resume — just reattach
(or start a new session) and rerun the same `python -u optuna_driver.py`
command from `/home/<user>/mace-finetune/cv-optuna`. It reloads
`optuna_study.log`, skips folds already in `fold_cache.json`, and only reruns
whatever was interrupted.

### 4. Stopping
The driver stops on its own after `MAX_TRIALS` (20) trials, or earlier if
results plateau (see stop conditions at the top of `optuna_driver.py`). On
exit it prints the best trial's number, params, and per-fold RMSEs — check
`driver.log` if you weren't watching the session live.

### 5. Retrieve results
Pull the study log back for `optuna_post-processing.ipynb`:

```bash
scp <user>@qbc1:/home/<user>/mace-finetune/cv-optuna/optuna_study.log mace-mh-1/cv-optuna/cv-results/optuna_study.log
```