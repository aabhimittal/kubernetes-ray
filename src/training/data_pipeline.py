"""
Zero-Copy Data Pipeline with Shared-Memory Optimization

Innovation: On multi-GPU nodes, data loaded once into a shared-memory
region (Ray's Plasma object store / POSIX /dev/shm) is read by all local
worker processes without per-worker copies. This eliminates the classic
N-copies-per-node overhead of naive DataLoaders.

Analogy: Instead of photocopying the same textbook for every student in a
room (one copy per GPU), you put one copy on a shared table everyone reads
from. The book (tensor) never moves; only pointers are handed out.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the zero-copy data pipeline."""

    batch_size: int = 32
    prefetch_batches: int = 2
    use_shared_memory: bool = True
    drop_last: bool = False


class SharedMemoryBatch:
    """
    A batch backed by a shared-memory buffer.

    When Ray is available the underlying numpy array is placed in the Plasma
    object store, which is memory-mapped and therefore zero-copy for readers
    on the same node. Without Ray we fall back to a plain numpy array (still
    zero-copy within a process via views).
    """

    def __init__(self, array: np.ndarray, object_ref=None):
        self._array = array
        self._object_ref = object_ref

    @property
    def object_ref(self):
        """Ray ObjectRef for cross-process zero-copy access (or None)."""
        return self._object_ref

    def as_array(self) -> np.ndarray:
        """Return the batch as a numpy array (a view, not a copy)."""
        return self._array

    def __len__(self) -> int:
        return len(self._array)


class ZeroCopyDataPipeline:
    """
    A minimal, framework-agnostic data pipeline that shards a dataset and
    yields shared-memory batches.

    The pipeline is deliberately dependency-light: it accepts any indexable
    dataset (``__len__`` + ``__getitem__``) and an optional collate function.
    """

    def __init__(
        self,
        dataset: Sequence,
        config: Optional[PipelineConfig] = None,
        collate_fn: Optional[Callable[[List], np.ndarray]] = None,
    ):
        self.dataset = dataset
        self.config = config or PipelineConfig()
        self.collate_fn = collate_fn or self._default_collate
        self._ray = self._maybe_import_ray()

    @staticmethod
    def _maybe_import_ray():
        try:
            import ray

            if ray.is_initialized():
                return ray
        except Exception:  # noqa: BLE001 - ray optional
            return None
        return None

    @staticmethod
    def _default_collate(samples: List) -> np.ndarray:
        """Stack samples into a single contiguous numpy array."""
        return np.ascontiguousarray(np.stack([np.asarray(s) for s in samples]))

    def _shard_indices(self, rank: int, world_size: int) -> List[int]:
        """Return the indices this rank is responsible for (strided shard)."""
        if world_size <= 1:
            return list(range(len(self.dataset)))
        return list(range(rank, len(self.dataset), world_size))

    def _make_batch(self, samples: List) -> SharedMemoryBatch:
        array = self.collate_fn(samples)
        if self.config.use_shared_memory and self._ray is not None:
            # Plasma store => memory-mapped, zero-copy for same-node readers.
            object_ref = self._ray.put(array)
            materialized = self._ray.get(object_ref)  # zero-copy view
            return SharedMemoryBatch(materialized, object_ref=object_ref)
        return SharedMemoryBatch(array)

    def iter_batches(
        self, rank: int = 0, world_size: int = 1
    ) -> Iterator[SharedMemoryBatch]:
        """
        Iterate over batches for a given data-parallel rank.

        Args:
            rank: This worker's rank in [0, world_size).
            world_size: Total number of data-parallel workers.
        """
        indices = self._shard_indices(rank, world_size)
        batch: List = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.config.batch_size:
                yield self._make_batch(batch)
                batch = []

        if batch and not self.config.drop_last:
            yield self._make_batch(batch)

    def __iter__(self) -> Iterator[SharedMemoryBatch]:
        return self.iter_batches()
