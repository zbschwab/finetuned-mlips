# Training Data

| File | Description |
|---|---|
| `dft-data/` | Gas-phase DFT INCARs/OUTCARs, used to compute E0s (the isolated-atom reference energy for each element — MACE subtracts per-atom offset). |
| `replay-data-mh-1-omat-pbe.xyz` | Replay dataset released alongside MACE-MH-1. |
| `EO_Project_reactivity_frames.xyz` | Selected configs from end of data-processing pipeline, filtered to {Ag, Ni, C, H, O}. From Box folder `EO_Project_reactivity`. |
| `O2DissociationSAA_2_frames.xyz` | Same pipeline, filtered to {Ag, C, H, O}. From Box folder `O2DissociationSAA_2`. |
| `train.xyz` | Combined data from all `*_frames.xyz` files |
| `selected_configs.xyz` | An element-filtered subsample of the replay dataset, output by `mace.cli.fine_tuning_select` |
| `selected_configs_combined.xyz` | Combined `selected_configs.xyz` and `train.xyz`, also output by `mace.cli.fine_tuning_select`. Not used in this project. |
| `test.xyz`, `train.xyz` | Test/train split: 10% held out as `test.xyz`; the remaining 90% is saved as `train.xyz`. |
| `folds/` | 8 files (train/valid (90/10 split) × 4 folds) for CV. |