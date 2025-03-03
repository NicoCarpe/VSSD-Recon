"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from multiprocessing import Pool

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]


def save_single_reconstruction(out_dir: Path, fname: str, recons: np.ndarray):
    """
    Save a single reconstruction to an h5 file.

    Args:
        out_dir: Path to the output directory where the reconstructions should be saved.
        fname: The filename under which the reconstruction should be saved.
        recons: The reconstruction data to be saved.
    """
    with h5py.File(out_dir / fname, 'w') as hf:
        hf.create_dataset("reconstruction", data=recons)

def save_reconstructions_mp(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images using multiprocessing.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard and uses multiprocessing to speed up the saving process.

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare data for multiprocessing
    args = [(out_dir, fname, recons) for fname, recons in reconstructions.items()]
    
    # Create a pool of processes
    with Pool() as pool:
        pool.starmap(save_single_reconstruction, args)


def save_reconstructions(reconstructions: Dict[str, np.ndarray], num_slc_dict, out_dir: Path):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        out_dir.mkdir(exist_ok=True, parents=True)
        # make sure folder of  out_dir / fname exists
        file_path = out_dir / fname
        file_path.parent.mkdir(exist_ok=True, parents=True) # in case fname also contains folders
        
        if fname in num_slc_dict:
            t_z, h, w = recons.shape
            recons = recons.reshape(t_z//num_slc_dict[fname], num_slc_dict[fname], h, w)
        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("reconstruction", data=recons)

def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data

def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                data[k] = v[()]
            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data

def load_shape(filename):
    """
    Load the shape of a .mat file.
    """
    with h5py.File(filename, 'r') as hf:
        key = list(hf.keys())[0]
        shape = hf[key].shape
    return shape

def load_mask(filename):
    """
    Load a mask from a .mat file.
    """
    data = loadmat(filename)
    keys = list(data.keys())[0]
    mask = data[keys]
    return mask

def load_kdata(filename):
    '''
    load kdata from .mat file
    return shape: [t,nz,nc,ny,nx]
    '''
    data = loadmat(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    kdata = kdata['real'] + 1j*kdata['imag']
    return kdata

