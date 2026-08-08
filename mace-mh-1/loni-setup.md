# LONI setup

Create these directories before your first run:

```bash
mkdir -p /home/<user>/mace-finetune/scripts
mkdir -p /home/<user>/mace-finetune/cv-optuna
mkdir -p /work/<user>/mace-finetune/data/folds
mkdir -p /work/<user>/mace-finetune/logs/mace_cv_search
```

Rule of thumb: anything you need to survive long-term (scripts, the Optuna study log) goes under `/home`; anything regenerable or high-volume (checkpoints, SLURM `.out`/`.err`, fold data) goes under `/work`.

## Conda environment

```bash
conda env create -f mace-mh-1/mace-mh-1.yml -p /work/<user>/.conda/envs/mace-mh-1
conda activate /work/<user>/.conda/envs/mace-mh-1
```

Environment lives on `/work`, not `/home` — it's large and regenerable from the `.yml`, so it doesn't belong in the backed-up, quota-limited tree.

## SLURM

`sbatch`/`sacct` exist only on the login node (`qbc1`) — compute nodes do not have SLURM client binaries. Any script that submits or polls jobs (`submit_and_wait()`, `optuna_driver.py`) must run from the login node, not as a job itself.

```bash
which sbatch   # /usr/local/bin/sbatch
which sacct    # /usr/bin/sacct
```

## tmux (CV/Optuna path only)

`optuna_driver.py` runs for the full duration of the search (potentially hours), polling SLURM in a loop. It must run in a `tmux` session on the login node — not as a SLURM job itself, and not in a plain terminal that dies on disconnect.

Direct finetuning does not need tmux: `full_train.sh` submits one `sbatch` job and returns immediately.

```bash
tmux new -s optuna
conda activate /work/<user>/.conda/envs/mace-mh-1
cd /home/<user>/mace-finetune/cv-optuna
python -u optuna_driver.py 2>&1 | tee driver.log
# detach: Ctrl-b d
# reattach: tmux attach -t optuna
```

Monitor from a **separate** terminal — don't `tail -f` inside the tmux session:

```bash
tail -f /home/<user>/mace-finetune/cv-optuna/driver.log
```

Job status:

```bash
squeue -u <user>
sacct -X -u <user>   # -X restricts to main job allocation, excludes .batch/.extern steps
```