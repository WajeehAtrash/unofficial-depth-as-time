"""Recovering a depthwise mixing schedule (Section 3.2 and Appendix B.2 of the paper).

For one-step generation the clean decode of block ``i`` is modeled as a mixture of the final
prediction and the input noise, with one coefficient pair shared by all trajectories::

    x_0^{i,n} = alpha_i * x_0^{L,n} + sigma_i * eps^n + delta_i^n

The least-squares problem is solved in closed form from sufficient statistics (inner products)
that are accumulated batch by batch, so memory does not grow with the number of trajectories
or with the image resolution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from depth_as_time.probe import LayerwiseProbe
from depth_as_time.task import TransportTask

_BASELINE_RELATIVE_TOLERANCE = 1e-9


@dataclass
class MixingStatistics:
    """Inner products needed to fit and score the two-component mixing model.

    With ``x`` the layer-``i`` decode, ``f`` the final decode and ``e`` the input noise, all
    sums run over trajectories and every pixel. ``xx``, ``xf`` and ``xe`` are per layer;
    ``ff``, ``fe`` and ``ee`` do not depend on the layer.
    """

    num_layers: int
    count: int = 0
    xx: Tensor = field(init=False)
    xf: Tensor = field(init=False)
    xe: Tensor = field(init=False)
    ff: float = 0.0
    fe: float = 0.0
    ee: float = 0.0

    def __post_init__(self) -> None:
        """Allocate the per-layer accumulators."""
        self.xx = torch.zeros(self.num_layers, dtype=torch.float64)
        self.xf = torch.zeros(self.num_layers, dtype=torch.float64)
        self.xe = torch.zeros(self.num_layers, dtype=torch.float64)

    def update(self, layers: Tensor, final: Tensor, noise: Tensor) -> None:
        """Add a batch of trajectories.

        Args:
            layers: Clean decodes of every layer, shape ``(L, B, *S)``.
            final: Final clean decode ``x_0^L``, shape ``(B, *S)``.
            noise: Input noise ``eps``, shape ``(B, *S)``.
        """
        if layers.shape[0] != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {layers.shape[0]}")
        f = final.flatten(1).double()
        e = noise.flatten(1).double()
        self.ff += (f * f).sum().item()
        self.fe += (f * e).sum().item()
        self.ee += (e * e).sum().item()
        for i in range(self.num_layers):
            x = layers[i].flatten(1).double()
            self.xx[i] += (x * x).sum().cpu()
            self.xf[i] += (x * f).sum().cpu()
            self.xe[i] += (x * e).sum().cpu()
        self.count += final.shape[0]

    def baseline_error(self) -> Tensor:
        """Squared error of predicting the final output directly, ``sum ||x - f||^2``."""
        return self.xx - 2.0 * self.xf + self.ff

    def reconstruction_error(self, alpha: Tensor, sigma: Tensor) -> Tensor:
        """Squared error ``sum ||x - alpha f - sigma e||^2`` of the two-component mixture."""
        return (
            self.xx
            - 2.0 * alpha * self.xf
            - 2.0 * sigma * self.xe
            + alpha**2 * self.ff
            + 2.0 * alpha * sigma * self.fe
            + sigma**2 * self.ee
        )


@dataclass(frozen=True)
class MixingResult:
    """Recovered depthwise schedule and its held-out reconstruction error.

    Attributes:
        alpha: Weight of the final prediction per layer, shape ``(L,)``.
        sigma: Weight of the input noise per layer, shape ``(L,)``.
        held_out_r: ``R_i`` on held-out trajectories per layer, shape ``(L,)``. ``R_i = 1``
            means no gain over predicting the final output. The last layer is NaN because its
            baseline error is zero by construction.
        num_train: Number of trajectories used for fitting.
        num_test: Number of held-out trajectories.
    """

    alpha: Tensor
    sigma: Tensor
    held_out_r: Tensor
    num_train: int
    num_test: int

    @property
    def mean_held_out_r(self) -> float:
        """Mean ``R_i`` over the layers where it is defined."""
        return torch.nanmean(self.held_out_r).item()


def fit_mixing_coefficients(train: MixingStatistics) -> tuple[Tensor, Tensor]:
    """Solve the 2x2 normal equations of ``min ||x - alpha f - sigma e||^2`` for every layer."""
    gram = torch.tensor([[train.ff, train.fe], [train.fe, train.ee]], dtype=torch.float64)
    rhs = torch.stack([train.xf, train.xe], dim=-1)  # (L, 2)
    coefficients = rhs @ torch.linalg.pinv(gram)  # gram is symmetric
    return coefficients[:, 0], coefficients[:, 1]


def evaluate_mixing(train: MixingStatistics, test: MixingStatistics) -> MixingResult:
    """Fit coefficients on ``train`` and report ``R_i`` on ``test``."""
    alpha, sigma = fit_mixing_coefficients(train)
    baseline = test.baseline_error()
    residual = test.reconstruction_error(alpha, sigma)
    defined = baseline > _BASELINE_RELATIVE_TOLERANCE * max(test.ff, 1e-30)
    ratio = torch.where(defined, residual / baseline.clamp_min(1e-300), torch.nan)
    return MixingResult(
        alpha=alpha.float(),
        sigma=sigma.float(),
        held_out_r=ratio.float(),
        num_train=train.count,
        num_test=test.count,
    )


@torch.no_grad()
def run_mixing_analysis(
    probe: LayerwiseProbe,
    noise_shape: tuple[int, ...],
    num_trajectories: int = 256,
    batch_size: int = 32,
    train_fraction: float = 0.7,
    num_classes: int = 0,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    on_batch: Callable[[Tensor], None] | None = None,
) -> MixingResult:
    """Run one-step generation on fresh noise and recover the depthwise mixing schedule.

    The paper uses 256 trajectories split 70/30 into 179 fitting and 77 held-out trajectories.

    Args:
        probe: Layerwise probe of the model under study.
        noise_shape: Shape ``S`` of a single input, e.g. ``(3, 256, 256)``.
        num_trajectories: Total number of trajectories ``N``.
        batch_size: Trajectories per network evaluation.
        train_fraction: Fraction of trajectories used for fitting.
        num_classes: If positive, labels are drawn uniformly from this many classes.
        seed: Seed for noise and labels. The same seed gives the same split.
        device: Device for the model evaluation.
        dtype: Dtype of the input noise, which must match the model weights (e.g. bfloat16).
        on_batch: Optional callback called before each batch with the global trajectory
            indices of that batch, shape ``(B,)``. Use it to set per-batch conditioning, for
            example the text prompt of a text-to-image model.
    """
    generator = torch.Generator().manual_seed(seed)
    num_train = int(train_fraction * num_trajectories)
    num_layers = probe.model.num_layers
    train, test = MixingStatistics(num_layers), MixingStatistics(num_layers)

    for start in range(0, num_trajectories, batch_size):
        size = min(batch_size, num_trajectories - start)
        noise = torch.randn(size, *noise_shape, generator=generator).to(device=device, dtype=dtype)
        label = None
        if num_classes > 0:
            label = torch.randint(0, num_classes, (size,), generator=generator).to(device)

        if on_batch is not None:
            on_batch(torch.arange(start, start + size))
        task = TransportTask.one_step(size, device=device, dtype=noise.dtype)
        clean = probe.run(noise, task, label).clean
        is_train = (torch.arange(start, start + size) < num_train).to(device)

        for stats, mask in ((train, is_train), (test, ~is_train)):
            if mask.any():
                stats.update(clean[:, mask], clean[-1, mask], noise[mask])

    return evaluate_mixing(train, test)
