# LONI setup

Create these directories before your first run:

```bash
mkdir -p /home/<user>/mace-finetune/scripts
mkdir -p /home/<user>/mace-finetune/cv-optuna
mkdir -p /work/<user>/mace-finetune/data
mkdir -p /work/<user>/mace-finetune/logs/cv-search
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
conda activate /work/<user>/.conda/envs/mace-mh-1
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