#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -A loni_mlips01
#SBATCH -o /work/zschwab/mace-finetune/logs/test_eval_%j.out
#SBATCH -e /work/zschwab/mace-finetune/logs/test_eval_%j.err

# $1 = TEST_XYZ, $2 = MODEL_PATH, $3 = OUTPUT_JSON

set -euo pipefail
date

TEST_XYZ=$1
MODEL_PATH=$2
OUTPUT_JSON=$3

module load conda/23.11.0
source /usr/local/packages/conda/23.11.0/etc/profile.d/conda.sh
conda activate /work/zschwab/.conda/envs/mace-mh-1

python /home/zschwab/mace-finetune/test_eval.py "${TEST_XYZ}" "${MODEL_PATH}" "${OUTPUT_JSON}"
date