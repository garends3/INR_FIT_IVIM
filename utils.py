import numpy as np
import nibabel as nib
import yaml
from pathlib import Path
from numpy.typing import ArrayLike


def parse_cfg(cfg_path: Path) -> dict[any, any]:
    """
    Parses the yaml file to a dictionary
    Args:
        cfg_path (Path): Path instance that points to the configuration yaml file
    Returns:
        dict[any, any]: parsed yaml file
    """
    with open(cfg_path, "r", encoding="utf8") as cfgyaml:
        cfg = yaml.safe_load(cfgyaml)
    return cfg


def parse_bvals_col(path: Path) -> np.array:
    return np.loadtxt(path, dtype=float)


def save_bvals(bvals: np.ndarray, output_path: Path) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    bval_strs = bvals.astype(str)
    with open(output_path, "w") as f:
        f.writelines([" ".join(bval_strs)])


def select_bvals(bval_path: Path, output_path: Path, selected_grads: ArrayLike) -> None:
    with open(bval_path, "r") as f:
        x = f.readline()

    sel_x = [val for i, val in enumerate(x.split()) if i in selected_grads]
    new_x = " ".join(sel_x)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with open(output_path, "w") as f:
        f.write(new_x)


def select_dwi_img(
    dwi_path: Path, output_path: Path, selected_grads: ArrayLike
) -> None:
    nifti_img = nib.load(dwi_path)
    img_data = nifti_img.get_fdata()[..., selected_grads]

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    sel_img = nib.Nifti1Image(
        img_data, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(sel_img, output_path)
