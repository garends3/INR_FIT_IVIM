import numpy as np
import torch

import signal_models


class IVIM:
    """Computes the predicted DWI signal from voxel-wise IVIM parameters
    (f, D, D*, S0) using the biexponential IVIM signal model.
    """

    def __init__(
        self,
        bvals: np.ndarray,
        device: str = "cpu",
    ):
        self.device = device
        self.bvals = torch.tensor(bvals, device=device)

    def compute_signal_from_coeff(self, coeffs: torch.tensor):
        IVIM_model = signal_models.IVIM()

        signal = IVIM_model(
            self.bvals.to(self.device),
            S0=coeffs[:, [-1]],
            D=coeffs[:, [-3]],
            Dstar=coeffs[:, [-2]],
            f=coeffs[:, [-4]],
        )

        return signal

    def compute_negative_signal(self, coeffs: torch.Tensor, **kwargs):
        return None
