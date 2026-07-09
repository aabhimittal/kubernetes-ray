"""
LLM Fine-Tuning Example

Shows how to configure the stack for a large-model workload: bf16 precision,
gradient checkpointing, hierarchical checkpointing, and the adaptive allocator
learning the best GPU count/strategy across runs.

This example is illustrative and runs on CPU with a synthetic loss curve, so
you can exercise the orchestration without a GPU cluster. Swap `train_step`
for a real forward/backward over your transformer to make it production-grade.

Run:
    python -m examples.llm_training
"""

import logging
import math

from src.core.gpu_allocator import AdaptiveGPUAllocator, WorkloadProfile
from src.training.distributed_trainer import DistributedTrainer, TrainingConfig
from src.training.mixed_precision import PrecisionConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("llm_example")


class DummyLLM:
    """Placeholder for a large transformer; only needs state_dict for ckpt."""

    def __init__(self, num_params: int):
        self.num_params = num_params

    def state_dict(self):
        return {"num_params": self.num_params}


class DummyOptimizer:
    def state_dict(self):
        return {"type": "AdamW"}


def main():
    workload = WorkloadProfile(
        model_size=7_000_000_000,  # 7B params
        batch_size=4,
        sequence_length=4096,
        precision="bf16",
        gradient_checkpointing=True,
    )

    # Persist bandit state across runs so allocation improves over time.
    allocator = AdaptiveGPUAllocator(max_gpus=8, state_file="llm_allocator_state.pkl")

    config = TrainingConfig(
        workload=workload,
        max_gpus=8,
        available_gpus=8,
        epochs=1,
        steps_per_epoch=50,
        checkpoint_every=1000,
        compute_capability=(9, 0),  # Hopper -> bf16 path
        precision=PrecisionConfig(preferred_dtype="bf16"),
    )

    model = DummyLLM(workload.model_size)
    optimizer = DummyOptimizer()

    def train_step(model, optimizer, batch, step):
        # Synthetic decaying loss to emulate convergence.
        loss = 4.0 * math.exp(-step / 20.0) + 1.5
        return {"loss": loss, "perplexity": math.exp(loss)}

    trainer = DistributedTrainer(config, allocator=allocator)
    result = trainer.train(model, optimizer, train_step)

    logger.info(
        "Fine-tune complete: loss=%.3f, ppl=%.1f on %d GPU(s) [%s]",
        result.final_metrics["loss"],
        result.final_metrics.get("perplexity", float("nan")),
        result.num_gpus,
        result.strategy,
    )

    best = allocator.get_best_allocation(workload, available_gpus=8)
    logger.info("Best known allocation so far: %s", best)


if __name__ == "__main__":
    main()
