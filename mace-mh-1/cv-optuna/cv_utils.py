"""
cv_utils.py

Helper functions for the Optuna + k-fold CV driver: submitting SLURM jobs
and parsing MACE training logs for RMSE metrics.
"""

import re
import subprocess
import time

EPOCH_PATTERN = re.compile(
    r"(?:Epoch (\d+)|(Initial)): head: (\S+), "
    r"loss=([\d.]+), "
    r"RMSE_E_per_atom=\s*([\d.]+) meV, "
    r"RMSE_F=\s*([\d.]+) meV / A, "
)

# check this
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
}


def parse_log(log_path, head="Default"):
    """Parse log from MACE run_train and return final epoch metrics (loss, RMSE_E, RMSE_F).

    Args:
        log_path: path to .out file to parse
        head: model finetune head name (default: "Default")

    Returns:
        dict (or None if no match found):
            epoch (int or None): Epoch number, or None for the "Initial" line.
            head (str): Matched head name.
            loss (float): Training loss.
            RMSE_E_per_atom (float): Energy RMSE, meV/atom.
            RMSE_F (float): Force RMSE, meV/A.
    """
    last_match = None

    with open(log_path, "r") as f:
        for line in f:
            m = EPOCH_PATTERN.search(line)
            if m and m.group(3) == head:
                last_match = m

    if last_match is None:
        return None

    # _ ignores first line of metrics tagged as "Initial" instead of "Epoch #"
    epoch_num, _, matched_head, loss, rmse_e, rmse_f = last_match.groups()

    return {
        "epoch": int(epoch_num) if epoch_num is not None else None,
        "head": matched_head,
        "loss": float(loss),
        "RMSE_E_per_atom": float(rmse_e),
        "RMSE_F": float(rmse_f),
    }


def submit_and_wait(
    script_path,
    log_dir,
    job_args=None,
    log_pattern="cv_test_{job_id}.out",
    poll_interval=30,
    timeout=None,
):
    """Submit sbatch script, then poll sacct to see log output until job terminates.

    Args:
        script_path: path to sbatch script to submit
        log_dir: directory containing the .out log (matches #SBATCH -o in script)
        job_args: list of positional args passed to sbatch script
        log_pattern: log filename pattern w/ job_id placeholder
        poll_interval: secs. between sacct polls
        timeout: (optional) max secs. to wait before terminating

    Returns:
        tuple: (str: job_id, str: state, str: log_path)
    """
    # submit sbatch job to LONI
    cmd = ["sbatch", script_path] + [str(a) for a in (job_args or [])]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # check for correct sbatch stdout: "Submitted batch job __"
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        raise RuntimeError(
            f"Could not parse job ID from sbatch output: {result.stdout!r}"
        )
    job_id = m.group(1)
    log_path = f"{log_dir}/{log_pattern.format(job_id=job_id)}"

    # poll log output from LONI with sacct
    start_time = time.time()
    while True:
        if timeout is not None and (time.time() - start_time) > timeout:
            raise TimeoutError(
                f"Job {job_id} did not reach a terminal state within {timeout}s"
            )

        sacct_result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "-X",
                "--format=State",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # -X flag means only show output relevant to the job allocation itself,
        # not internal steps like .batch or .extern

        line = sacct_result.stdout.strip()
        state = (
            line.split()[0] if line else None
        )  # strip trailing "CANCELLED by <uid>" text

        if state and any(state.startswith(t) for t in TERMINAL_STATES):
            return job_id, state, log_path

        time.sleep(poll_interval)


def restart_job(script_path, log_dir, max_retries=2, **kwargs):
    """Submit a job, resubmitting on TIMEOUT up to max_retries.
    Each resubmission resumes from the last MACE checkpoint.

    Raises:
        RuntimeError: if the job is still TIMEOUT after max_retries resubmits.

    Returns:
        tuple: (job_id, state, log_path) from final attempt.
    """
    job_id, state, log_path = None, None, None
    for attempt in range(max_retries + 1):
        job_id, state, log_path = submit_and_wait(script_path, log_dir, **kwargs)
        if state != "TIMEOUT":
            return job_id, state, log_path
    raise RuntimeError(
        f"Job {job_id} still TIMEOUT after {max_retries} retries (script={script_path})"
    )
