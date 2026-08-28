# INR_FIT_IVIM

Voxel-wise IVIM (intravoxel incoherent motion) fitting using an implicit
neural representation (INR). Instead of fitting each voxel independently,
a small coordinate-based MLP is trained to map spatial position directly
to the IVIM parameters (f, D, D*) and S0, using the acquired diffusion-
weighted signal as supervision.

Code accompanying: Arends, Gerrit & Fitski, Matthijs et al.
"Direct correlation of intravoxel incoherent motion imaging based on implicit neural representations and histopathology in pediatric Wilms’ tumors: can it differentiate between subtypes?."

Code is derived from fitting algorithms for https://github.com/tomhend/MSMT-CSD_INR ((MSMT-)CSD) and https://github.com/tomhend/Standard_model_INR/tree/main (Standard Model of White Matter)

## Installation

Create the conda environment from `environment.yml`:

```
conda env create -f environment.yml
conda activate ivim_inr
```

## Usage

Run from the command line, pointing at a configuration file:

```
python main.py -c <config file path>
```

Or edit the default config path in `main.py` and run without the `-c`
flag (e.g. from an IDE).


## Configuration

See `configs/example_config_IVIM_kidney.yaml` for a fully documented
example, and `configs/WT_IVIM_kidney_all.yaml` for the configuration used
to produce the results in the article. Key options:

- `loss_function_name`: `mselossavgl2noneg` (MSE loss) or `ricianlogloss`
  (Rician log-likelihood loss, requires an `s_noise` estimate).
- `dataset_name` / `output_calculator` / `model_name`: all set to `IVIM`.
- `scale_data`: normalizes the signal before fitting (recommended for
  training stability); `norm` scales by the 95th percentile of the b0
  signal.
- `gaussian_encoding`: Gaussian Fourier feature positional encoding for
  the input coordinates (recommended).
- `lr_scheduler`: enables a step learning-rate scheduler (decay rate is
  hard-coded in `main.py`).
- `train_cfg.lpos` / `sigma`: number and frequency spread of positional
  encodings.
- `train_cfg.n_layers` / `hidden_dim`: INR network size.
- `train_cfg.lr` / `epochs` / `batch_size`: standard training
  hyperparameters.

## Outputs

For each run, `main.py` writes to `paths.output/<experiment_name>/`:

- `model_<experiment_name>.pt`: trained model weights (best training loss).
- `params_<experiment_name>.nii.gz`: fitted IVIM parameter maps, channel
  order `[f, D, D*, S0]`.
- `pred_signal_<experiment_name>.nii.gz`: the model's reconstructed DWI
  signal, for comparison against the input data.

## Repository layout

- `main.py` / `main_optuna.py`: entry points for training / hyperparameter
  search.
- `models.py`: the INR architecture (`IVIM`).
- `datasets.py`: loads a DWI nifti + bval file into a per-voxel dataset.
- `signal_models.py` / `diffusion_calculator.py`: the biexponential IVIM
  signal equation and the glue that turns network output into predicted
  signal.
- `loss_functions.py`, `ml_utils.py` (training loop), `utils.py` (I/O
  helpers), 

## License

GPLv3, see `LICENSE`.
