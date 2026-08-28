import torch


class IVIM:
    """Biexponential IVIM signal model.

    S(b) = S0 * (f * exp(-b * D*) + (1 - f) * exp(-b * D))
    """

    def __init__(self):
        self.param_names = ['f', 'D', 'Dstar']

    def __call__(self, bvals, S0, D, Dstar, f):
        S = S0 * (f * torch.exp(-bvals * Dstar) + (1 - f) * torch.exp(-bvals * D))

        return S
