#!/bin/bash
#SBATCH -N 1
#SBATCH -p single
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 168:00:00
#SBATCH -A loni_mlips01
#SBATCH -o /work/zschwab/mace-finetune/logs/driver_%j.out
#SBATCH -e /work/zschwab/mace-finetune/logs/driver_%j.err

export PATH="/usr/local/bin:$PATH"

module load conda/23.11.0
source /usr/local/packages/conda/23.11.0/etc/profile.d/conda.sh
conda activate /work/zschwab/.conda/envs/mace-mh-1

cd /home/zschwab/mace-finetune/cv-optuna
python -u optuna_driver.py