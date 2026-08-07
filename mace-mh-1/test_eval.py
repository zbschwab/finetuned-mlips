"""
test_eval.py

Single-point MLIP predictions on test set (train_withNi.xyz) vs. DFT reference.
Run once on the held-out test set (test.xyz) only.

Run this script with the paths to your test set, model, and desired output 
location on LONI:

sbatch test_eval.sh \
    /work/zschwab/mace-finetune/data/test.xyz \
    /work/zschwab/mace-finetune/final-run-740875/checkpoints/full_run_run-123_stagetwo.model \
    /work/zschwab/mace-finetune/final-run-740875/results/test_eval.json
"""

import json
import sys
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def evaluate(test_xyz, model_path, output_json):
    configs = read(test_xyz, index=":")
    calc = MACECalculator(
        model_paths=model_path, device="cuda", default_dtype="float32", head="default"
    )

    e_errs, f_errs = [], []
    per_config = []

    for i, atoms in enumerate(configs):
        atoms.calc = calc
        e_pred = atoms.get_potential_energy()
        f_pred = atoms.get_forces()

        e_ref = atoms.info["REF_energy"]
        f_ref = atoms.arrays["REF_forces"]

        e_err = (e_pred - e_ref) / len(atoms)  # per-atom
        f_err = (f_pred - f_ref).flatten()
        f_rmse = float(np.sqrt(np.mean(f_err**2)) * 1000)

        e_errs.append(e_err)
        f_errs.append(f_err)

        per_config.append(
            {
                "index": i,
                "formula": atoms.get_chemical_formula(),
                "n_atoms": len(atoms),
                "E_error_meV_per_atom": float(e_err * 1000),
                "F_RMSE_meV_per_A": f_rmse,
            }
        )

    e_errs = np.array(e_errs)
    f_errs = np.concatenate(f_errs)

    results = {
        "n_configs": len(configs),
        "RMSE_E_per_atom_meV": float(np.sqrt(np.mean(e_errs**2)) * 1000),
        "RMSE_F_meV_per_A": float(np.sqrt(np.mean(f_errs**2)) * 1000),
        "per_config": per_config,
    }
    print({k: v for k, v in results.items() if k != "per_config"})
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
