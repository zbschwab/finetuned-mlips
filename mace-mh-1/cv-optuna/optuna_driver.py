"""
optuna_driver.py

TPE (Tree-structured Parzen Estimator) search over 4-fold CV for MACE multihead finetuning.
Completed trials (one 4-fold run) cached in `optuna_study.log`.

Resumable across kills at any point, including mid-trial:
    - fold_cache.json caches each completed fold, keyed by trial number.
    - resume_orphaned_trials() finishes any trial left RUNNING by a prior
    kill by reconstructing it from storage (already-sampled params are
    replayed, not resampled) and re-running objective() on it, so cached
    folds skip straight through and only the interrupted fold onward
    actually reruns.
    - replay_stop_state() rebuilds the noise-floor/no-improvement counters
    from full trial history on every startup, so warm-up and stop
    conditions behave the same whether this is trial 1 of a fresh study
    or trial 1 of the Nth resume.
"""

import statistics
import os
import optuna
from cv_utils import parse_log, restart_job, get_fold_result, save_fold_result

STUDY_NAME = "mace_cv_search"

HOME_DIR = "/home/zschwab/mace-finetune"
WORK_DIR = "/work/zschwab/mace-finetune"

SCRIPT = f"{HOME_DIR}/scripts/fold_train.sh"
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
WEIGHT_E = 0.65
WEIGHT_F = 0.35


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
        trial: optuna.Trial for this run, passed in by study.optimize()
            or reconstructed by resume_orphaned_trials()

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
                trial_id=trial.number,
                fold_id=fold_id,
                job_args=job_args,
                log_pattern="mace_{job_id}.out",  # must match #SBATCH -o in fold_train.sh
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
    fold_combined = [WEIGHT_E*e + WEIGHT_F*f for e, f in zip(fold_rmse_e, fold_rmse_f)]
    trial.set_user_attr("fold_std", statistics.stdev(fold_combined) if len(fold_combined) > 1 else 0.0)
    combined = WEIGHT_E * statistics.mean(fold_rmse_e) + WEIGHT_F * statistics.mean(fold_rmse_f)
    return combined


def resume_orphaned_trials(study):
    """Finish any trial left RUNNING by a prior kill (dropped tmux session,
    SIGKILL, etc.) instead of leaving it stuck in storage indefinitely.

    A RUNNING trial already has its hyperparameters persisted in
    optuna_study.log from when they were first sampled. Reconstructing a
    Trial from its storage id and calling objective() on it again replays
    those same params rather than resampling, and any fold already in
    fold_cache.json is skipped, so this only redoes the work that was
    actually interrupted.

    Caution: if a fold's SLURM job was still queued or running on the
    cluster at the moment of the kill, this will submit a second job for
    that fold. Check `squeue` before resuming if you're unsure.
    """
    orphans = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    for frozen in orphans:
        trial = optuna.trial.Trial(study, frozen._trial_id)
        print(f"resuming orphaned trial #{trial.number}")
        try:
            value = objective(trial)
        except optuna.TrialPruned as e:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            print(f"  trial #{trial.number} pruned on resume: {e}")
            continue
        except Exception as e:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"  trial #{trial.number} failed on resume: {e}")
            continue
        study.tell(trial, value)
        print(f"  trial #{trial.number} completed on resume: {value:.4f}")


def replay_stop_state(completed_trials):
    """Reconstruct prev_best / no_improve / noise_streak from already-completed
    trial history, so a resumed run doesn't discard evidence of convergence
    gathered in earlier sessions.

    Args:
        completed_trials: list of optuna.trial.FrozenTrial with
            state == COMPLETE, any order.

    Returns:
        tuple: (prev_best: float, no_improve: int, noise_streak: int)
    """
    prev_best = float("inf")
    no_improve = 0
    noise_streak = 0

    for t in sorted(completed_trials, key=lambda t: t.number):
        value = t.value
        fold_std = t.user_attrs.get("fold_std", 0.0)
        improved = value < prev_best
        delta = prev_best - value if improved else 0.0

        if improved:
            prev_best = value
            no_improve = 0
        else:
            no_improve += 1

        noise_streak = noise_streak + 1 if (improved and 0 < delta < fold_std) else 0

    return prev_best, no_improve, noise_streak


def main():
    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(STUDY_LOG))
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=123),  # default n_startup_trials=10
        direction="minimize",
        load_if_exists=True,
    )

    # heal any trial left RUNNING by prior kill so history replay reflects true state
    resume_orphaned_trials(study)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    prev_best, no_improve, noise_streak = replay_stop_state(completed)

    # Count all completed trials across every past session (via `completed`,
    # from full study history) so it's correct after killing/restarting sessions
    session_trial_count = len(completed)

    for _ in range(MAX_TRIALS):
        study.optimize(objective, n_trials=1, catch=(optuna.TrialPruned,))
        trial = study.trials[-1]
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        session_trial_count += 1

        value = trial.value
        fold_std = trial.user_attrs.get("fold_std", 0.0)
        improved = value < prev_best
        delta = prev_best - value if improved else 0.0

        if improved:
            prev_best = value
            no_improve = 0
        else:
            no_improve += 1

        if session_trial_count <= N_STARTUP_TRIALS:
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
