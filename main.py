import argparse
import time
from pathlib import Path

import torch
import nibabel as nib

from utils import parse_cfg

from loss_functions import get_loss_function
from output_calculators import get_output_calculator
from ml_utils import Trainer
from models import get_model
from datasets import get_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def initialize_run(cfg_path=None):
    if cfg_path is not None:
        cfg = parse_cfg(Path(cfg_path))
    else:
        cfg = parse_cfg(Path("configs/WT_IVIM_kidney_all.yaml"))

    train_cfg = cfg["train_cfg"]

    width = cfg["width"]
    height = cfg["height"]
    depth = cfg["depth"]

    lr = train_cfg["lr"]
    lambda_ = train_cfg["lambda"]
    log_freq = cfg["log_freq"]

    model = get_model(cfg)
    print(model)

    dataset = get_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=3,
        drop_last=True,
    )

    loss_fn = get_loss_function(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        StepLR(optimizer, step_size=1, gamma=0.99) if cfg["lr_scheduler"] else None
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epochs = train_cfg["epochs"]

    output_calculator = get_output_calculator(cfg, dataset=dataset, device=device)

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        data_shape=(width, height, depth),
        output_calculator=output_calculator,
        log_freq=log_freq,
        lambda_=lambda_,
        scheduler=scheduler,
        clip_grads=cfg["clip_gradients"],
    )

    start_time = time.time()
    best_model, _ = trainer.train()
    print(f"Training finished in {time.time() - start_time:.1f} s")

    file_inf = cfg.get("experiment_name", "nameless")
    output_folder = Path(cfg["paths"]["output"]) / file_inf

    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    torch.save(best_model, output_folder / f"model_{file_inf}.pt")

    nifti_img, pred_signal_img, param_maps = trainer.create_full_output_image(
        cfg, dataset.get_scale()
    )

    # param_maps channel order: f, D, D*, S0 (see models.IVIM.forward)
    param_maps_img = nib.Nifti1Image(
        param_maps, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(param_maps_img, output_folder / f"params_{file_inf}.nii.gz")

    pred_signal_nifti = nib.Nifti1Image(
        pred_signal_img, affine=nifti_img.affine, header=nifti_img.header
    )
    nib.save(pred_signal_nifti, output_folder / f"pred_signal_{file_inf}.nii.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="IVIM INR",
        description="Fits the IVIM model voxel-wise using an implicit neural representation",
    )
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args = parser.parse_args()

    initialize_run(args.config)
