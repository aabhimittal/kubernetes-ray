"""
Computer Vision Training Example

Demonstrates driving the DistributedTrainer with a small synthetic image
classification workload. It runs anywhere (CPU-only included) because the
trainer and its components degrade gracefully when Ray/torch/GPUs are absent.

Run:
    python -m examples.vision_training
"""

import logging

import numpy as np

from src.core.gpu_allocator import WorkloadProfile
from src.monitoring.metrics_collector import MetricsCollector
from src.training.data_pipeline import PipelineConfig, ZeroCopyDataPipeline
from src.training.distributed_trainer import DistributedTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vision_example")


class TinyLinearModel:
    """A dependency-free linear classifier (numpy) standing in for a CNN."""

    def __init__(self, in_dim: int, num_classes: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(scale=0.01, size=(in_dim, num_classes))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    def state_dict(self):
        return {"weights": self.weights.copy()}


class SGDOptimizer:
    """Minimal SGD to keep the example self-contained."""

    def __init__(self, model: TinyLinearModel, lr: float = 0.01):
        self.model = model
        self.lr = lr

    def apply_gradient(self, grad: np.ndarray):
        self.model.weights -= self.lr * grad

    def state_dict(self):
        return {"lr": self.lr}


def build_dataset(n: int = 256, dim: int = 64, num_classes: int = 10):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(n, dim)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=n)
    return list(zip(features, labels))


def main():
    dim, num_classes = 64, 10
    dataset = build_dataset(dim=dim, num_classes=num_classes)
    labels = np.array([label for _, label in dataset])

    # Feed the trainer real (features, labels) batches through the zero-copy
    # pipeline so a single shared-memory buffer backs each batch.
    feature_pipeline = ZeroCopyDataPipeline(
        [f for f, _ in dataset], PipelineConfig(batch_size=32, drop_last=True)
    )
    label_pipeline = ZeroCopyDataPipeline(
        labels.tolist(), PipelineConfig(batch_size=32, drop_last=True)
    )
    batches = [
        (fb.as_array(), np.asarray(lb.as_array()))
        for fb, lb in zip(feature_pipeline, label_pipeline)
    ]

    model = TinyLinearModel(dim, num_classes)
    optimizer = SGDOptimizer(model, lr=0.05)
    metrics = MetricsCollector()

    def train_step(model, optimizer, batch, step):
        x, y = batch

        logits = model.forward(x)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = float(-np.log(probs[np.arange(len(y)), y] + 1e-9).mean())

        onehot = np.eye(logits.shape[1])[y]
        grad = x.T @ (probs - onehot) / len(y)
        optimizer.apply_gradient(grad)

        metrics.record_training_stats(throughput=1.0, loss=loss, allocated_gpus=1)
        return {"loss": loss}

    workload = WorkloadProfile(
        model_size=dim * num_classes,
        batch_size=32,
        sequence_length=1,
        precision="fp16",
        gradient_checkpointing=False,
    )
    config = TrainingConfig(
        workload=workload,
        max_gpus=4,
        available_gpus=4,
        epochs=2,
        steps_per_epoch=20,
        checkpoint_every=1000,
        compute_capability=(7, 5),  # Turing -> fp16 path
    )

    trainer = DistributedTrainer(config)
    result = trainer.train(model, optimizer, train_step, data_iterable=batches * 2)

    logger.info("Final loss: %.4f", result.final_metrics["loss"])
    logger.info(
        "Ran %d steps on %d GPU(s), strategy=%s in %.2fs",
        result.steps_completed,
        result.num_gpus,
        result.strategy,
        result.wall_time_seconds,
    )
    logger.info("Sample metrics:\n%s", metrics.render())


if __name__ == "__main__":
    main()
