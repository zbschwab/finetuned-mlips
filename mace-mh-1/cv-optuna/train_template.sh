#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -A loni_mlips01
#SBATCH -o /work/zschwab/mace-finetune/logs/mace_%j.out
#SBATCH -e /work/zschwab/mace-finetune/logs/mace_%j.err
#SBATCH --mail-user zschwab@tulane.edu
#SBATCH --mail-type ALL

# Positional args (passed via submit_and_wait()):
#   $1 = TRIAL_ID          (Optuna trial.number, fixed across resubmits/folds)
#   $2 = FOLD_ID
#   $3 = FOLD_TRAIN_FILE   (e.g. data/folds/fold0_train.xyz)
#   $4 = FOLD_VALID_FILE   (e.g. data/folds/fold0_valid.xyz)
#   $5 = LR
#   $6 = WEIGHT_DECAY
#   $7 = SWA_LR
#   $8 = SWA_ENERGY_WEIGHT
#   $9 = SWA_FORCES_WEIGHT

set -euo pipefail
date

TRIAL_ID=$1
FOLD_ID=$2
FOLD_TRAIN_FILE=$3
FOLD_VALID_FILE=$4
LR=$5
WEIGHT_DECAY=$6
SWA_LR=$7
SWA_ENERGY_WEIGHT=$8
SWA_FORCES_WEIGHT=$9

ENERGY_WEIGHT=1.0
FORCES_WEIGHT=100.0

export HOME_DIR=/home/$USER/mace-finetune
export WORK_DIR=/work/$USER/mace-finetune

mkdir -p "$WORK_DIR/data"
cp -r "$HOME_DIR"/data/. "$WORK_DIR"/data/
cd "$WORK_DIR"

module load conda/23.11.0
source /usr/local/packages/conda/23.11.0/etc/profile.d/conda.sh
conda activate /work/zschwab/.conda/envs/mace-mh-1

# resubmit after TIMEOUT reuses same dir and --restart_latest finds existing checkpoint
RUN_NAME="t${TRIAL_ID}_f${FOLD_ID}"
RUN_DIR="$WORK_DIR/run-${RUN_NAME}"
mkdir -p "$RUN_DIR"/{models,checkpoints,logs,results}


python -m mace.cli.run_train \
--name "${RUN_NAME}" \
--restart_latest \
--train_file "data/${FOLD_TRAIN_FILE}" \
--valid_file "data/${FOLD_VALID_FILE}" \
--foundation_model mh-1 \
--foundation_head omat_pbe \
--pt_train_file data/selected_configs_withNi.xyz \
--work_dir "$RUN_DIR/models" \
--model_dir "$RUN_DIR/models" \
--log_dir "$RUN_DIR/logs" \
--checkpoints_dir "$RUN_DIR/checkpoints" \
--results_dir "$RUN_DIR/results" \
--energy_weight "${ENERGY_WEIGHT}" \
--forces_weight "${FORCES_WEIGHT}" \
--weight_decay "${WEIGHT_DECAY}" \
--swa \
--start_swa 20 \
--swa_energy_weight "${SWA_ENERGY_WEIGHT}" \
--swa_forces_weight "${SWA_FORCES_WEIGHT}" \
--stress_weight 0.0 \
--lr "${LR}" \
--swa_lr "${SWA_LR}" \
--multiheads_finetuning True \
--E0s "{47: -0.19458509, 28: -0.59150255, 8: -1.55774442, 6: -1.28369203, 1: -1.11167391}" \
--atomic_numbers "[1, 6, 8, 28, 47]" \
--max_num_epochs 30 \
--device "cuda"

status=$?
date
exit $status