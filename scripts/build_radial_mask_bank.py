import os
from os.path import join
import re
import glob
import argparse
import h5py
import numpy as np
from mri_utils import load_mask


def parse_acceleration_from_filename(filename):
    """
    Given a basename like 'flow2d_mask_ktRadial8.mat', extract the integer
    acceleration. Matches digits immediately after 'Radial' and before '.mat'.
    Returns int or None.
    """
    match = re.search(r"Radial(\d+)\.mat$", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def build_radial_mask_bank(mat_folder, output_h5_path):
    """
    1) Finds all files under `mat_folder` matching '*Radial*.mat' in
       '*/TrainingSet/Mask_TaskAll/*/*/P*/' hierarchy.
    2) Parses each filename to get its acceleration factor (integer).
    3) Loads each .mat via load_mask():
         – If 2D, treat as shape (nx, ny) → expand to (1, nx, ny).
         – If 3D, assume shape is already (nt, nx, ny).
       Keep that as raw_3d = (nt, nx, ny).
    4) Keeps only the file with the largest nt for each unique (acc, nx, ny).
    5) Ensures the output folder exists, then writes each “best” mask into an HDF5
       under the key 'acc{acceleration}_{nx}x{ny}', storing data as (nt, nx, ny).
    """
    # 1) Build the glob pattern
    pattern = join(
        mat_folder,
        "*/TrainingSet/Mask_TaskAll/*/*/P*/*Radial*.mat"
    )
    file_list = sorted(glob.glob(pattern))

    if len(file_list) == 0:
        raise FileNotFoundError(f"No files found matching pattern:\n  {pattern}")

    # Map (acc, width, height) → (best_filepath, best_nt)
    best_per_combo = {}

    for filepath in file_list:
        basename = os.path.basename(filepath)
        acc = parse_acceleration_from_filename(basename)
        if acc is None:
            print(f"Skipping (cannot parse acceleration): {basename}")
            continue

        # 3) Load the mask array via load_mask()
        try:
            raw = load_mask(filepath)
        except Exception as e:
            print(f"  ▶ Error loading '{basename}': {e}")
            continue

        # If 2D, expand to (1, nx, ny); if 3D, assume (nt, nx, ny)
        if raw.ndim == 2:
            raw_3d = raw[np.newaxis, :, :]  # shape = (1, nx, ny)
        elif raw.ndim == 3:
            raw_3d = raw  # shape = (nt, nx, ny)
        else:
            print(f"  ▶ '{basename}' has invalid shape {raw.shape}; skipping.")
            continue

        nt, nx, ny = raw_3d.shape
        key = (acc, nx, ny)
        if key not in best_per_combo or nt > best_per_combo[key][1]:
            best_per_combo[key] = (filepath, nt)

    if not best_per_combo:
        raise RuntimeError("No valid (acceleration, width, height) combinations were found.")

    #  ‣ Ensure output directory exists before creating HDF5
    out_dir = os.path.dirname(output_h5_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 5) Write each “best” mask into the HDF5 as (nt, nx, ny)
    with h5py.File(output_h5_path, "w") as h5f:
        for (acc, nx, ny), (best_path, best_nt) in best_per_combo.items():
            basename = os.path.basename(best_path)

            # Reload and ensure shape (nt, nx, ny)
            try:
                raw = load_mask(best_path)
            except Exception as e:
                print(f"  ▶ Could not reload '{basename}' for HDF5 write: {e}")
                continue

            if raw.ndim == 2:
                raw_3d = raw[np.newaxis, :, :]
            else:
                raw_3d = raw  # assume (nt, nx, ny)

            if raw_3d.shape != (best_nt, nx, ny):
                print(f"  ▶ Warning: '{basename}' shape {raw_3d.shape} != expected ({best_nt}, {nx}, {ny}); skipping.")
                continue

            dataset_name = f"acc{acc}_{nx}x{ny}"
            h5f.create_dataset(dataset_name, data=raw_3d, compression="gzip")
            print(f"  • Wrote   '{dataset_name}'  ← {basename}  (nt={best_nt})")

    print(f"\nDone. All largest‐slice masks saved to '{output_h5_path}'.")
    print(f"Total unique (acceleration, width, height) combos: {len(best_per_combo)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Gather all Radial‐named .mat masks, assume 3D files come in as (nt, nx, ny), "
            "expand any 2D→(1, nx, ny), pick the largest nt per (acc, nx, ny), "
            "and save into one HDF5 as (nt, nx, ny)."
        )
    )
    parser.add_argument(
        "--mat_folder",
        type=str,
        required=True,
        help="Base folder containing '*/TrainingSet/Mask_TaskAll/.../P*/...Radial*.mat'."
    )
    parser.add_argument(
        "--out_h5",
        type=str,
        default="radial_masks.h5",
        help="Path to the output .h5 file (default: radial_masks.h5)."
    )
    args = parser.parse_args()

    build_radial_mask_bank(args.mat_folder, args.out_h5)
