from typing import Protocol, Any
from torch import Tensor

from diffusion_calculator import IVIM
from datasets import IVIMDataset


class OutputCalculator(Protocol):
    def output_from_model_out(self, model_out: Any, **kwargs) -> Any:
        ...


class CoeffDiff:
    def __init__(self, diff_calculator: Any):
        self.diff_calculator = diff_calculator

    def output_from_model_out(self, model_out: Any, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        diff_signal = self.diff_calculator.compute_signal_from_coeff(model_out, **kwargs)
        neg_signal = self.diff_calculator.compute_negative_signal(model_out, **kwargs)
        return diff_signal, model_out, neg_signal


def create_IVIM_calculator(
    cfg: dict, dataset: IVIMDataset, device: str, **kwargs
) -> CoeffDiff:
    diff_calculator = IVIM(
        dataset.get_bvals(),
        device=device,
    )
    return CoeffDiff(diff_calculator)


OUTPUT_CALCULATORS = {
    "IVIM": create_IVIM_calculator,
}


def get_output_calculator(cfg: dict, **kwargs) -> OutputCalculator:
    constructor = None
    if cfg.get("output_calculator", None):
        constructor = OUTPUT_CALCULATORS.get(cfg["output_calculator"], None)

    if constructor is None:
        raise Exception("No output calculator for found")
    return constructor(cfg, **kwargs)
