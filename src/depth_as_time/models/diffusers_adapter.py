"""Generic adapter that exposes a diffusers MMDiT transformer as a :class:`LayerwiseModel`.

It works for any diffusers transformer that follows the common layout

* one or more ``nn.ModuleList`` attributes holding the blocks, run in order,
* a final ``norm_out(hidden_states, temb)`` followed by ``proj_out``,

which covers ``FluxTransformer2DModel`` (FLUX.1 schnell and dev) and ``SD3Transformer2DModel``
(SD3, SD3.5 and their turbo variants). Blocks are captured with forward hooks, so the
network runs exactly as the library implements it. No block code is copied.

Only the image tokens are decoded. Text tokens that share the joint-attention stream are
dropped before the readout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from depth_as_time.models.base import LayerwiseModel
from depth_as_time.task import TransportTask


def unpatchify_nchw(tokens: Tensor, patch_size: int, height: int, width: int) -> Tensor:
    """Fold ``(B, N, p*p*C)`` patch tokens into ``(B, C, H, W)`` (the SD3 output layout)."""
    batch = tokens.shape[0]
    grid_h, grid_w = height // patch_size, width // patch_size
    channels = tokens.shape[-1] // (patch_size * patch_size)
    x = tokens.reshape(batch, grid_h, grid_w, patch_size, patch_size, channels)
    return torch.einsum("nhwpqc->nchpwq", x).reshape(batch, channels, height, width)


@dataclass(frozen=True)
class DiffusersFamilySpec:
    """How a model family maps the package conventions onto the diffusers forward call.

    Attributes:
        block_attrs: Names of the block lists in execution order.
        timestep_scale: The transformer expects ``timestep = timestep_scale * t``. FLUX scales
            by 1000 internally and takes ``t`` in ``[0, 1]``; SD3 expects ``1000 * t``.
        unpatchify_output: If true, the head output is folded back to ``(B, C, H, W)``.
            FLUX works on packed tokens and its pipeline does the unpacking, so it is false.
    """

    block_attrs: tuple[str, ...]
    timestep_scale: float
    unpatchify_output: bool


FLUX_SPEC = DiffusersFamilySpec(
    block_attrs=("transformer_blocks", "single_transformer_blocks"),
    timestep_scale=1.0,
    unpatchify_output=False,
)
SD3_SPEC = DiffusersFamilySpec(
    block_attrs=("transformer_blocks",),
    timestep_scale=1000.0,
    unpatchify_output=True,
)

FAMILY_SPECS: dict[str, DiffusersFamilySpec] = {"flux": FLUX_SPEC, "sd3": SD3_SPEC}

# Conditioning tensors with a leading batch dimension. Position ids (``txt_ids``, ``img_ids``) are
# shared across the batch and are passed through untouched.
_BATCHED_CONDITIONING_KEYS = ("encoder_hidden_states", "pooled_projections", "guidance")


def infer_family(transformer: nn.Module) -> str:
    """Guess the family key from the transformer class name."""
    name = type(transformer).__name__.lower()
    if "flux" in name:
        return "flux"
    if "sd3" in name:
        return "sd3"
    raise ValueError(
        f"Cannot infer the model family from {type(transformer).__name__}. "
        "Pass an explicit DiffusersFamilySpec."
    )


@dataclass
class _Capture:
    """Mutable state filled by the hooks during one network call."""

    image_tokens: int = 0
    latent_hw: tuple[int, int] | None = None
    states: list[Tensor] = field(default_factory=list)
    temb: Tensor | None = None


class DiffusersTransformerAdapter(LayerwiseModel):
    """Wrap a diffusers transformer so its blocks can be probed.

    Text conditioning is fixed per adapter through :meth:`set_conditioning`, because it does
    not depend on the transport task. The class ``label`` argument of the probing API is
    unused.

    Args:
        transformer: A loaded diffusers transformer, in ``eval()`` mode.
        spec: Family specification. Inferred from the class name when ``None``.
        block_attrs: Optional subset of the block lists to probe, for example
            ``("single_transformer_blocks",)`` to reproduce the paper's FLUX analysis over the
            38 single-stream blocks.
    """

    def __init__(
        self,
        transformer: nn.Module,
        spec: DiffusersFamilySpec | None = None,
        block_attrs: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.spec = spec if spec is not None else FAMILY_SPECS[infer_family(transformer)]
        self.block_attrs = tuple(block_attrs) if block_attrs is not None else self.spec.block_attrs
        self._blocks = [block for attr in self.block_attrs for block in getattr(transformer, attr)]
        if not self._blocks:
            raise ValueError(f"No blocks found in attributes {self.block_attrs}")
        self._conditioning: dict[str, Any] = {}
        self._capture = _Capture()
        self._handles = self._register_hooks()

    @property
    def num_layers(self) -> int:
        """Number of probed blocks across all selected block lists."""
        return len(self._blocks)

    def set_conditioning(self, **conditioning: Any) -> None:
        """Set keyword arguments forwarded to the transformer on every call.

        Tensors with a leading batch dimension of 1 are broadcast to the batch size of ``z``,
        so one prompt can be paired with many noise draws.

        Examples:
            FLUX: ``encoder_hidden_states``, ``pooled_projections``, ``txt_ids``, ``img_ids``
            and, for guidance-distilled variants, ``guidance``.
            SD3: ``encoder_hidden_states``, ``pooled_projections``.
        """
        self._conditioning = conditioning

    def _conditioning_for_batch(self, batch_size: int) -> dict[str, Any]:
        resolved = {}
        for key, value in self._conditioning.items():
            if key in _BATCHED_CONDITIONING_KEYS and torch.is_tensor(value):
                if value.shape[0] == 1:
                    value = value.expand(batch_size, *value.shape[1:])
                elif value.shape[0] != batch_size:
                    raise ValueError(
                        f"Conditioning {key!r} has batch size {value.shape[0]} but the input has "
                        f"{batch_size}. Use batch size 1 to broadcast."
                    )
            resolved[key] = value
        return resolved

    def close(self) -> None:
        """Remove all hooks from the wrapped transformer."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _register_hooks(self) -> list[torch.utils.hooks.RemovableHandle]:
        handles = [
            block.register_forward_hook(self._capture_block, with_kwargs=True)
            for block in self._blocks
        ]
        handles.append(self.transformer.norm_out.register_forward_pre_hook(self._capture_temb))
        return handles

    def _capture_block(self, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> None:
        self._capture.states.append(self._image_stream(output))

    def _capture_temb(self, module: nn.Module, args: tuple) -> None:
        # norm_out is called as norm_out(hidden_states, temb)
        self._capture.temb = args[1]

    def _image_stream(self, output: Any) -> Tensor:
        """Extract the image tokens from a block output.

        Blocks return either ``(text_tokens, image_tokens)`` or, in older single-stream
        implementations, one tensor with the text tokens in front of the image tokens.
        """
        n = self._capture.image_tokens
        if isinstance(output, tuple):
            return output[-1]
        return output[:, -n:]

    @torch.no_grad()
    def hidden_states(
        self, z: Tensor, task: TransportTask, label: Tensor | None = None
    ) -> list[Tensor]:
        """Run the transformer once and return the image tokens after every probed block."""
        self._capture = self._new_capture(z)
        timestep = task.t * self.spec.timestep_scale
        self.transformer(
            hidden_states=z,
            timestep=timestep,
            return_dict=False,
            **self._conditioning_for_batch(z.shape[0]),
        )
        states = self._capture.states
        if len(states) != self.num_layers:
            raise RuntimeError(
                f"Captured {len(states)} block outputs but expected {self.num_layers}. "
                "A block may have been skipped or called more than once."
            )
        return states

    def readout_tokens(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        """Apply ``norm_out`` and ``proj_out`` to image tokens, reusing the captured ``temb``."""
        temb = self._capture.temb
        if temb is None:
            raise RuntimeError(
                "Call hidden_states() before readout so the conditioning is captured."
            )
        return self.transformer.proj_out(self.transformer.norm_out(h, temb))

    def tokens_to_output(self, tokens: Tensor) -> Tensor:
        """Fold SD3 patch tokens to a latent image. FLUX tokens are returned unchanged."""
        if not self.spec.unpatchify_output:
            return tokens
        if self._capture.latent_hw is None:
            raise RuntimeError("Call hidden_states() before tokens_to_output().")
        return unpatchify_nchw(tokens, self.transformer.config.patch_size, *self._capture.latent_hw)

    def _new_capture(self, z: Tensor) -> _Capture:
        """Record how many image tokens ``z`` contains so text tokens can be dropped."""
        if z.ndim == 3:  # packed tokens (B, N, C), as used by FLUX
            return _Capture(image_tokens=z.shape[1])
        patch = self.transformer.config.patch_size
        height, width = z.shape[-2:]
        return _Capture(
            image_tokens=(height // patch) * (width // patch), latent_hw=(height, width)
        )
