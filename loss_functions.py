from typing import Callable
from functools import partial
import torch.nn


def avg_mse_l2_loss_noneg(
    output: torch.Tensor,
    labels: torch.Tensor,
    coeffs: torch.Tensor,
    neg_values: torch.Tensor,
    _lambda: torch.Tensor,
    *args,
    **kwargs
) -> torch.Tensor:
    loss = ((output - labels) ** 2).sum(dim=1).mean()
    return loss


def rician_log_loss(
        output: torch.Tensor,
        labels: torch.Tensor,
        coeffs: torch.Tensor,
        neg_values: torch.Tensor,
        _lambda: torch.Tensor,
        s_noise: torch.Tensor,
        *args,
        **kwargs
) -> torch.Tensor:
    # Rician loss

    term1 = torch.log(labels / (s_noise ** 2))

    term2 = -(labels ** 2 + output ** 2) / (2 * (s_noise ** 2))

    z = (labels * output) / (s_noise ** 2)
    I0e = torch.special.i0e(z)
    lI0e = torch.log(I0e)
    term3 = lI0e + z

    log_pdf = term1 + term2 + term3

    n_batch = labels.shape[0]
    loss = -torch.sum(log_pdf) / n_batch
    return loss


def create_avg_mse_l2noneg(cfg: dict, *args, **kwargs) -> Callable:
    _lambda = torch.tensor(cfg["train_cfg"]["lambda"], dtype=torch.float)
    return partial(avg_mse_l2_loss_noneg, _lambda=_lambda)


def create_rician_log_loss(cfg: dict, *args, **kwargs) -> Callable:
    _lambda = torch.tensor(cfg["train_cfg"]["lambda"], dtype=torch.float)
    s_noise = torch.tensor(cfg["s_noise"], dtype=torch.float)
    return partial(rician_log_loss, _lambda=_lambda, s_noise=s_noise)


LOSS_FUNCTIONS = {
    "mselossavgl2noneg": create_avg_mse_l2noneg,
    "ricianlogloss": create_rician_log_loss,
}


def get_loss_function(cfg: dict, *args, **kwargs) -> Callable:
    constructor = LOSS_FUNCTIONS.get(cfg["loss_function_name"], None)
    if constructor is None:
        raise Exception("Loss function name not recognized")
    return constructor(cfg, args, kwargs)
