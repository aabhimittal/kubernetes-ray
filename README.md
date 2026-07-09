# Ray + Kubernetes GPU Scaling

**Enterprise-grade distributed GPU training infrastructure** built on
[Ray](https://www.ray.io/) and [Kubernetes](https://kubernetes.io/), with a set
of novel optimizations for adaptive resource allocation, fault tolerance, and
load balancing.

[![CI](https://github.com/OWNER/kubernetes-ray/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/kubernetes-ray/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why this project?

Training large models on GPUs at scale is hard: nodes fail, GPUs sit idle,
data pipelines bottleneck, and picking the right number of GPUs for a given
workload is guesswork. This project packages battle-tested patterns behind a
small, framework-light Python API so you can focus on your model.

### The Orchestra Analogy

Think of distributed GPU training like a symphony orchestra:

| Component | Role |
|-----------|------|
| **Ray** | The conductor coordinating all musicians (GPUs) |
| **Kubernetes** | The concert hall that can expand/contract seating |
| **GPUs** | Individual musicians, each specialized but needing coordination |
| **Training data** | Sheet music distributed efficiently to each player |
| **Checkpoints** | Recordings so you can resume if a musician drops out |

---

## Novel optimizations

1. **Adaptive GPU Allocation** — a Thompson-Sampling multi-armed bandit learns,
   per workload fingerprint, which `(num_gpus, placement_strategy)` combination
   maximizes throughput. It balances exploration and exploitation automatically.
   → `src/core/gpu_allocator.py`

2. **Hierarchical Fault Tolerance** — four-tier checkpointing (shared memory →
   local SSD → network storage → object storage), each with its own frequency
   and retention, plus recovery that falls back down the hierarchy.
   → `src/core/checkpoint_manager.py`

3. **Zero-Copy Data Pipeline** — batches land in Ray's Plasma object store
   (memory-mapped shared memory), so every worker on a node reads one copy
   instead of `N` copies.
   → `src/training/data_pipeline.py`

4. **Quantum-Inspired Scheduling** — task-to-GPU assignment via simulated
   annealing (a classical emulation of quantum annealing), minimizing an energy
   function over load imbalance and communication affinity.
   → `src/scheduling/quantum_scheduler.py`

5. **Hybrid Precision Training** — automatic fp16/bf16 selection by GPU compute
   capability, with a dynamic loss scaler for fp16 overflow handling.
   → `src/training/mixed_precision.py`

6. **Predictive Autoscaling** — combines a reactive utilization signal with an
   EWMA forecast of pending work, with cool-downs to prevent thrashing.
   → `src/scheduling/autoscaler.py`

---

## Project structure

```
kubernetes-ray/
├── src/
│   ├── core/
│   │   ├── ray_cluster.py          # Ray cluster management + GPU discovery
│   │   ├── gpu_allocator.py        # Adaptive (bandit) GPU allocation
│   │   └── checkpoint_manager.py   # Hierarchical checkpointing
│   ├── training/
│   │   ├── distributed_trainer.py  # End-to-end training orchestrator
│   │   ├── data_pipeline.py        # Zero-copy data loading
│   │   └── mixed_precision.py      # Hybrid precision training
│   ├── scheduling/
│   │   ├── quantum_scheduler.py    # Quantum-inspired load balancing
│   │   └── autoscaler.py           # Predictive K8s autoscaling logic
│   └── monitoring/
│       ├── metrics_collector.py    # Prometheus metrics
│       └── dashboard.py            # Metrics + health HTTP endpoint
├── k8s/
│   ├── ray-cluster.yaml            # KubeRay RayCluster + PVC
│   ├── gpu-operator.yaml           # Namespace, quota, NVIDIA runtime notes
│   ├── autoscaler.yaml             # HPA on GPU-utilization metrics
│   └── monitoring.yaml             # Prometheus + Grafana wiring
├── examples/
│   ├── vision_training.py          # Synthetic image-classification run
│   ├── llm_training.py             # LLM fine-tuning configuration
│   └── multimodal_training.py      # Scheduler + autoscaler showcase
├── tests/                          # Pytest suite (runs CPU-only)
├── docker/
│   ├── Dockerfile                  # CUDA runtime image
│   └── requirements.txt
├── setup.py / pyproject.toml
└── README.md / LICENSE
```

---

## Quick start

### Local (CPU-only) — runs anywhere

The library degrades gracefully when Ray, PyTorch, or GPUs are unavailable, so
you can explore the orchestration on a laptop.

```bash
git clone https://github.com/OWNER/kubernetes-ray.git
cd kubernetes-ray
pip install -e ".[dev]"

# Run the examples
python -m examples.vision_training
python -m examples.llm_training
python -m examples.multimodal_training

# Run the test suite
pytest -q
```

### Minimal training loop

```python
from src.core.gpu_allocator import WorkloadProfile
from src.training.distributed_trainer import DistributedTrainer, TrainingConfig

workload = WorkloadProfile(
    model_size=7_000_000_000, batch_size=4, sequence_length=4096,
    precision="bf16", gradient_checkpointing=True,
)
config = TrainingConfig(workload=workload, max_gpus=8, available_gpus=8,
                        epochs=1, steps_per_epoch=100)

def train_step(model, optimizer, batch, step):
    # your forward/backward here
    return {"loss": ...}

trainer = DistributedTrainer(config)
result = trainer.train(model, optimizer, train_step)
print(result)
```

### On Kubernetes (with GPUs)

```bash
# 1. Install prerequisites (once):
#    - NVIDIA GPU Operator (see k8s/gpu-operator.yaml header)
#    - KubeRay operator (see k8s/ray-cluster.yaml header)

# 2. Build & push the image:
docker build -f docker/Dockerfile -t ghcr.io/OWNER/ray-k8s-gpu-scaling:latest .
docker push ghcr.io/OWNER/ray-k8s-gpu-scaling:latest

# 3. Deploy:
kubectl apply -f k8s/gpu-operator.yaml   # namespace + quota
kubectl apply -f k8s/ray-cluster.yaml    # RayCluster + storage
kubectl apply -f k8s/monitoring.yaml     # Prometheus + Grafana
kubectl apply -f k8s/autoscaler.yaml     # HPA

# 4. Access the dashboards:
kubectl -n gpu-training port-forward svc/gpu-training-cluster-head-svc 8265:8265
```

---

## Monitoring

`src/monitoring/dashboard.py` serves three endpoints (default port `8266`):

- `GET /metrics` — Prometheus exposition of GPU utilization, throughput, loss,
  and allocated GPU count
- `GET /healthz` — Kubernetes liveness/readiness probe
- `GET /` — a minimal HTML summary of cluster + training state

These are scraped by the Prometheus config in `k8s/monitoring.yaml` and
visualized in Grafana alongside NVIDIA DCGM GPU metrics.

---

## Design notes

- **Framework-light by design.** The orchestrator takes a `train_step` callable,
  so it stays agnostic to your model architecture and even to PyTorch itself.
- **Testable without a cluster.** Every component has a graceful fallback when
  Ray / torch / CUDA are missing, and the full test suite runs on CPU.
- **Feedback loop.** Observed throughput is fed back into the bandit allocator,
  so allocation decisions improve across runs (state is persisted to disk).

---

## Development

```bash
pip install -e ".[dev]"
pip install ruff
ruff check src tests examples
pytest -q --cov=src
```

CI runs lint + tests across Python 3.9–3.11 (`.github/workflows/ci.yml`).

---

## License

[MIT](./LICENSE) © Abhishek Mittal
