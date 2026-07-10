"""Tests for the mixed-precision helpers and end-to-end trainer loop."""

from src.core.gpu_allocator import WorkloadProfile
from src.training.data_pipeline import PipelineConfig, ZeroCopyDataPipeline
from src.training.distributed_trainer import DistributedTrainer, TrainingConfig
from src.training.mixed_precision import (
    DynamicLossScaler,
    PrecisionConfig,
    select_dtype,
)


def test_select_dtype_by_compute_capability():
    assert select_dtype((8, 0)) == "bf16"  # Ampere+
    assert select_dtype((7, 5)) == "fp16"  # Turing
    assert select_dtype((7, 0), preferred="fp32") == "fp32"


def test_dynamic_loss_scaler_backoff_and_growth():
    scaler = DynamicLossScaler(PrecisionConfig(init_scale=1024.0, growth_interval=2))
    start = scaler.scale
    scaler.update(found_inf=True)
    assert scaler.scale < start  # backed off

    stable = scaler.scale
    scaler.update(found_inf=False)
    scaler.update(found_inf=False)  # hits growth interval
    assert scaler.scale > stable


def test_data_pipeline_batches_and_sharding():
    dataset = list(range(10))
    pipe = ZeroCopyDataPipeline(dataset, PipelineConfig(batch_size=4))
    batches = list(pipe.iter_batches(rank=0, world_size=1))
    # 10 items, batch 4 -> [4, 4, 2]
    assert [len(b) for b in batches] == [4, 4, 2]


def test_data_pipeline_sharding_is_disjoint():
    dataset = list(range(12))
    pipe = ZeroCopyDataPipeline(dataset, PipelineConfig(batch_size=2))
    rank0 = [int(v) for b in pipe.iter_batches(0, 2) for v in b.as_array()]
    rank1 = [int(v) for b in pipe.iter_batches(1, 2) for v in b.as_array()]
    assert set(rank0).isdisjoint(rank1)
    assert sorted(rank0 + rank1) == dataset


def test_end_to_end_training_loop():
    workload = WorkloadProfile(1000, 8, 128, "bf16", False)
    config = TrainingConfig(
        workload=workload,
        max_gpus=4,
        available_gpus=4,
        epochs=1,
        steps_per_epoch=10,
        checkpoint_every=1000,  # avoid disk writes in this test
    )
    trainer = DistributedTrainer(config)

    class _Model:
        def state_dict(self):
            return {}

    class _Opt:
        def state_dict(self):
            return {}

    losses = []

    def train_step(model, optimizer, batch, step):
        loss = 1.0 / (step + 1)
        losses.append(loss)
        return {"loss": loss}

    result = trainer.train(_Model(), _Opt(), train_step)
    assert result.steps_completed == 10
    assert result.num_gpus >= 1
    assert result.final_metrics["loss"] > 0
    assert len(losses) == 10
