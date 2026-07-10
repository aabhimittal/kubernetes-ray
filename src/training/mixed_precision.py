"""
Hybrid Precision Training with Adaptive Gradient Scaling

Innovation: A precision policy that picks fp16 vs bf16 based on the GPU's
compute capability, plus a dynamic loss scaler that backs off on overflow
and grows again after a stable window. This wraps torch.cuda.amp when
PyTorch is present but degrades gracefully so the logic stays unit-testable
on CPU-only machines.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional
    torch = None
    TORCH_AVAILABLE = False


@dataclass
class PrecisionConfig:
    """Configuration for mixed-precision training."""

    enabled: bool = True
    preferred_dtype: str = "auto"  # auto | fp16 | bf16 | fp32
    init_scale: float = 2.0 ** 16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000


def select_dtype(compute_capability: Tuple[int, int], preferred: str = "auto") -> str:
    """
    Choose a training dtype.

    bf16 is preferred on Ampere+ (compute capability >= 8.0) because it has
    the dynamic range of fp32 and rarely overflows. Older GPUs fall back to
    fp16, which needs loss scaling.
    """
    if preferred != "auto":
        return preferred
    major = compute_capability[0]
    return "bf16" if major >= 8 else "fp16"


class DynamicLossScaler:
    """
    A pure-Python dynamic loss scaler mirroring torch.cuda.amp.GradScaler's
    growth/backoff behaviour, usable without a GPU for testing.
    """

    def __init__(self, config: Optional[PrecisionConfig] = None):
        self.config = config or PrecisionConfig()
        self.scale = self.config.init_scale
        self._growth_tracker = 0

    def update(self, found_inf: bool) -> None:
        """
        Update the scale after a step.

        Args:
            found_inf: True if inf/NaN gradients were detected this step.
        """
        if found_inf:
            self.scale = max(1.0, self.scale * self.config.backoff_factor)
            self._growth_tracker = 0
            logger.debug("Gradient overflow: scale backed off to %.1f", self.scale)
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.config.growth_interval:
                self.scale *= self.config.growth_factor
                self._growth_tracker = 0
                logger.debug("Scale grown to %.1f", self.scale)


class MixedPrecisionTrainer:
    """
    A thin helper that wires together dtype selection, autocast, and a
    gradient scaler for a single training step.
    """

    def __init__(
        self,
        compute_capability: Tuple[int, int] = (8, 0),
        config: Optional[PrecisionConfig] = None,
    ):
        self.config = config or PrecisionConfig()
        self.dtype_name = select_dtype(compute_capability, self.config.preferred_dtype)
        self.scaler = DynamicLossScaler(self.config)

        # A real GradScaler is only needed for fp16; bf16 does not require it.
        self._torch_scaler = None
        if TORCH_AVAILABLE and self.config.enabled and self.dtype_name == "fp16":
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                self._torch_scaler = torch.cuda.amp.GradScaler(
                    init_scale=self.config.init_scale
                )

    @property
    def torch_dtype(self):
        """Return the corresponding torch dtype (or None if torch absent)."""
        if not TORCH_AVAILABLE:
            return None
        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[self.dtype_name]

    def autocast(self):
        """Return an autocast context manager (or a no-op if torch absent)."""
        if TORCH_AVAILABLE and self.config.enabled and self.dtype_name != "fp32":
            return torch.autocast(device_type="cuda", dtype=self.torch_dtype)
        return _NullContext()

    def step(self, loss, optimizer) -> None:
        """
        Perform a scaled backward + optimizer step.

        Falls back to a plain step when torch/GPU AMP is unavailable.
        """
        if self._torch_scaler is not None:
            self._torch_scaler.scale(loss).backward()
            self._torch_scaler.step(optimizer)
            self._torch_scaler.update()
        else:
            loss.backward()
            optimizer.step()


class _NullContext:
    """A do-nothing context manager used when autocast is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
