"""
optuna_driver.py

TPE (Tree-structured Parzen Estimator) search over 4-fold CV for MACE multihead finetuning.
Completed trials (one 4-fold run) cached in `optuna_study.log`, but no fold-level restart.
"""

import statistics
import os
import optuna
from cv_utils import parse_log, restart_job, get_fold_result, save_fold_result

STUDY_NAME = "mace_cv_search"

HOME_DIR = "/home/zschwab/mace-finetune"
WORK_DIR = "/work/zschwab/mace-finetune"

SCRIPT = f"{HOME_DIR}/scripts/train_template.sh"
LOG_DIR = f"{WORK_DIR}/logs/{STUDY_NAME}"
FOLD_DIR = f"{WORK_DIR}/data/folds"
STUDY_LOG = f"{HOME_DIR}/cv-optuna/optuna_study.log"
FOLD_CACHE = f"{HOME_DIR}/cv-optuna/fold_cache.json"
os.makedirs(LOG_DIR, exist_ok=True)

N_FOLDS = 4
MAX_RETRIES = 2
MAX_TRIALS = 20
NO_IMPROVE_LIMIT = 4  # only evaluated after N_STARTUP_TRIALS have completed (random phase)
NOISE_STREAK_REQUIRED = 3  # number of consecutive trials that must improve on prev_best
# by less than that trial's fold_std before we consider it to be a
# real plateau (vs. one lucky noise-sized delta) and stop
N_STARTUP_TRIALS = 10  # TPESampler default; ~2 random pts/dim across the 5-param space

# weights applied to mean fold RMSE_E and F to combine them into single scalar Optuna minimizes.
# prevents optimizing one at expense of the other. higher F b/c more sensitive to hyperparams than E
WEIGHT_E = 0.45
WEIGHT_F = 0.55


def objective(trial):
    """Run one Optuna trial: sample one hyperparameter combination, then
    evaluate it across all 4 CV folds (by submitting one SLURM job per fold
    via restart_job()). Each fold trains from foundation model.

    Sampled hyperparameters (see cv-optuna/README.md):
        lr: learning rate (log-uniform)
        weight_decay: L2 regularization (log-uniform)
        swa_lr: SWA learning rate (log-uniform)
        swa_energy_weight: SWA energy loss weight (uniform)
        swa_forces_weight: SWA forces loss weight (uniform)

    Args:
        trial: optuna.Trial for this run, passed in by study.optimize().

    Returns:
        float: weighted combination of mean fold RMSE_E and RMSE_F

    Raises:
        optuna.TrialPruned: if any fold fails, times out after retries, or
            its log can't be parsed for metrics.
    """
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 5e-7, log=True)
    swa_lr = trial.suggest_float("swa_lr", 1e-6, 1e-4, log=True)
    swa_energy_weight = trial.suggest_float("swa_energy_weight", 0.5, 10)
    swa_forces_weight = trial.suggest_float("swa_forces_weight", 50, 150)

    fold_rmse_f, fold_rmse_e = [], []

    for fold_id in range(N_FOLDS):
        cached = get_fold_result(trial.number, fold_id, FOLD_CACHE)
        if cached is not None:
            fold_rmse_f.append(cached["RMSE_F"])
            fold_rmse_e.append(cached["RMSE_E_per_atom"])
            continue

        job_args = [
            trial.number,
            fold_id,
            f"{FOLD_DIR}/fold{fold_id}_train.xyz",
            f"{FOLD_DIR}/fold{fold_id}_valid.xyz",
            lr,
            weight_decay,
            swa_lr,
            swa_energy_weight,
            swa_forces_weight,
        ]
        try:
            job_id, state, log_path = restart_job(
                SCRIPT,
                LOG_DIR,
                max_retries=MAX_RETRIES,
                job_args=job_args,
                log_pattern="mace_{job_id}.out",  # must match #SBATCH -o in train_template.sh
            )
        except RuntimeError as e:
            raise optuna.TrialPruned(f"fold {fold_id} failed after retries: {e}")

        if state != "COMPLETED":
            raise optuna.TrialPruned(f"fold {fold_id} ended in state {state}")

        metrics = parse_log(log_path, head="Default")
        if metrics is None:
            raise optuna.TrialPruned(f"fold {fold_id}: no metrics parsed from {log_path}")

        save_fold_result(
            trial.number, fold_id, metrics["RMSE_E_per_atom"], metrics["RMSE_F"], FOLD_CACHE
        )
        fold_rmse_f.append(metrics["RMSE_F"])
        fold_rmse_e.append(metrics["RMSE_E_per_atom"])

    trial.set_user_attr("fold_rmse_f", fold_rmse_f)
    trial.set_user_attr("fold_rmse_e", fold_rmse_e)
    trial.set_user_attr("fold_std", statistics.stdev(fold_rmse_f) if len(fold_rmse_f) > 1 else 0.0)
    combined = WEIGHT_E * statistics.mean(fold_rmse_e) + WEIGHT_F * statistics.mean(fold_rmse_f)
    return combined


def main():
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(STUDY_LOG))
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=123),  # default n_startup_trials=10
        direction="minimize",
        load_if_exists=True,
    )

    # seed state from any already-completed trials (handles resuming an Optuna study)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    prev_best = min((t.value for t in completed), default=float("inf"))
    no_improve = 0
    noise_streak = 0

    for _ in range(MAX_TRIALS):
        study.optimize(objective, n_trials=1, catch=(optuna.TrialPruned,))
        trial = study.trials[-1]
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        value = trial.value
        fold_std = trial.user_attrs.get("fold_std", 0.0)
        improved = value < prev_best
        delta = prev_best - value if improved else 0.0

        if improved:
            prev_best = value
            no_improve = 0
        else:
            no_improve += 1

        if trial.number < N_STARTUP_TRIALS:
            continue

        if improved and 0 < delta < fold_std:
            noise_streak += 1
        else:
            noise_streak = 0

        if noise_streak >= NOISE_STREAK_REQUIRED:
            print(f"stop: {noise_streak} consecutive trials within noise floor")
            break
        if no_improve >= NO_IMPROVE_LIMIT:
            print(f"stop: {no_improve} trials without improvement")
            break

    print(f"best trial #{study.best_trial.number}: combined={study.best_value:.4f}")
    print(f"  RMSE_F folds: {study.best_trial.user_attrs['fold_rmse_f']}")
    print(f"  RMSE_E folds: {study.best_trial.user_attrs['fold_rmse_e']}")
    print(f"best params: {study.best_params}")


if __name__ == "__main__":
    main()
