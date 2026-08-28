from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import parse_bvals_col


def create_input_space_prop(width: int, height: int, depth: int) -> torch.Tensor:
    """Builds a (width, height, depth, 3) grid of normalized coordinates in
    [-1, 1] that preserves the aspect ratio of the input volume (the
    longest axis spans the full [-1, 1] range, the others are scaled
    proportionally).
    """
    dims = np.array([width, height, depth])
    max_dim = np.max(dims)
    p_size = 2 / max_dim

    sizes = p_size * (dims - 1)

    x = torch.linspace(-sizes[0] / 2, sizes[0] / 2, width)
    y = torch.linspace(-sizes[1] / 2, sizes[1] / 2, height)
    z = torch.linspace(-sizes[2] / 2, sizes[2] / 2, depth)

    input_tensor = torch.cartesian_prod(x, y, z)

    return input_tensor.reshape(width, height, depth, 3)


class DiffusionDataset(Dataset):
    def get_bvals(self):
        pass

    def get_scale(self) -> float:
        pass

    def get_val_set(self):
        pass


class IVIMDataset(DiffusionDataset):
    """Loads a 4D DWI nifti (plus a bval file and an optional mask) and
    exposes it as a per-voxel (coordinate -> signal) dataset for
    INR-based IVIM fitting.
    """

    def __init__(
        self,
        bval_path: Path,
        bval_delta: float,
        nifti_path: Path,
        mask_path: Path = None,
        scale: str | int | float = 1,
        val_size: int = 0,
    ) -> None:
        nifti_file = nib.load(nifti_path)
        full_img = nifti_file.get_fdata()
        full_img[full_img < 1e-6] = 1e-6
        full_img = np.nan_to_num(full_img)

        self.bvals = parse_bvals_col(bval_path)

        voxel_wise_norm = False

        if isinstance(scale, int) or isinstance(scale, float):
            self.scale_values = scale
        elif scale == "voxel_norm":
            voxel_wise_norm = True
            b0_idx = self.bvals < bval_delta
            self.scale_values = np.percentile(
                full_img[..., b0_idx], 95, axis=-1, keepdims=True
            )

            self.scale_values[self.scale_values < 1] = 1
        elif scale == "norm":
            b0_idx = self.bvals < 0.0011
            self.scale_values = np.percentile(
                full_img[..., b0_idx], 95)

        width, height, depth, n_grad = full_img.shape
        self.input_tensor = create_input_space_prop(width, height, depth)

        output_array = full_img

        if mask_path:
            mask_data = nib.load(mask_path).get_fdata().astype(bool)
            mask_data = np.squeeze(mask_data)
            self.input_tensor = self.input_tensor[mask_data]
            output_array = output_array[mask_data]

            if voxel_wise_norm:
                self.scale_values = self.scale_values[mask_data]
        else:
            self.input_tensor = self.input_tensor.reshape(width * height * depth, 3)
            output_array = output_array.reshape(width * height * depth, n_grad)

        output_array = output_array / self.scale_values
        if val_size > 0:
            val_idx = np.random.choice(output_array.shape[0], output_array.shape[0]//100*val_size, replace=False)
            bool_arr = np.zeros(output_array.shape[0], dtype=bool)
            bool_arr[val_idx] = 1

            self.val_input = self.input_tensor[bool_arr, :]
            self.val_output = torch.tensor(output_array[bool_arr, :])

            self.input_tensor = self.input_tensor[~bool_arr, :]
            output_array = output_array[~bool_arr, :]
        else:
            self.val_input, self.val_output = None, None

        self.output_tensor = torch.tensor(output_array, dtype=torch.float)

    def __len__(self) -> int:
        return self.input_tensor.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.input_tensor[idx], self.output_tensor[idx], {}

    def get_val_set(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.val_input, self.val_output

    def get_bvals(self) -> np.ndarray:
        return self.bvals

    def get_scale(self) -> float:
        return self.scale_values


def create_IVIM(cfg: dict) -> IVIMDataset:
    mask_path = Path(cfg["paths"]["mask"]) if cfg["paths"]["mask"] else None

    dataset = IVIMDataset(
        bval_path=Path(cfg["paths"].get("fsl_bvals", None)),
        bval_delta=cfg["bval_delta"],
        nifti_path=Path(cfg["paths"]["nifti"]),
        mask_path=mask_path,
        scale=cfg["scale_data"],
        val_size=cfg.get("val_size", 0),
    )

    return dataset


DATASETS = {
    "IVIM": create_IVIM,
}


def get_dataset(cfg: dict) -> DiffusionDataset:
    constructor = DATASETS.get(cfg["dataset_name"], None)
    if constructor is None:
        raise Exception("Dataset name not recognized")
    return constructor(cfg)
