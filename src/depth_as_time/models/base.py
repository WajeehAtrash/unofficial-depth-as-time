"""Interface a network must expose so its layers can be probed."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from depth_as_time.task import TransportTask


class LayerwiseModel(nn.Module, ABC):
    """A one-step generator with a dimension-preserving residual stream (Eq. 1).

    The network computes ``h_0 = E(z)`` and ``h_{i+1} = B_{i+1}(h_i; tau_task, c)``, and its
    output is ``v = W_out(h_L; tau_task, c)``. To probe an existing network, wrap it in a
    subclass that implements :meth:`hidden_states` and :meth:`readout_tokens`.

    The velocity returned by the readout may be an instantaneous velocity or an
    interval-averaged one (MeanFlow). Both are decoded with the same rule (Eq. 3).
    """

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of blocks ``L``."""

    @abstractmethod
    def hidden_states(
        self, z: Tensor, task: TransportTask, label: Tensor | None = None
    ) -> list[Tensor]:
        """Run the network and return the residual stream after every block.

        Args:
            z: Network input ``z_t`` of shape ``(B, *S)``.
            task: Transport task conditioning.
            label: Optional class labels of shape ``(B,)``. ``None`` means unconditional.

        Returns:
            ``[h_1, ..., h_L]``. The embedding output ``h_0`` is not included.
        """

    @abstractmethod
    def readout_tokens(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        """Apply the output head ``W_out`` (normalization, AdaLN, linear) to a hidden state.

        The result stays in token space, e.g. ``(B, N, patch_dim)`` for a patch transformer.
        Models without a token structure return the final output directly.
        """

    def tokens_to_output(self, tokens: Tensor) -> Tensor:
        """Map head outputs to the input shape, e.g. unpatchify. Identity by default."""
        return tokens

    def readout(self, h: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        """The model's own readout ``W_out(h; tau_task, c)`` with the shape of the input."""
        return self.tokens_to_output(self.readout_tokens(h, task, label))

    def forward(self, z: Tensor, task: TransportTask, label: Tensor | None = None) -> Tensor:
        """Ordinary network output: the head applied to the last hidden state."""
        return self.readout(self.hidden_states(z, task, label)[-1], task, label)
