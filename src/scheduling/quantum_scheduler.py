"""
Quantum-Inspired Scheduler for GPU Workload Load Balancing

Innovation: Uses simulated annealing (a classical algorithm inspired by
quantum annealing) to solve the NP-hard task-to-GPU assignment problem.

The Physics Analogy:
    A quantum annealer finds the ground state of an Ising Hamiltonian by
    slowly reducing quantum fluctuations. We emulate this classically with
    simulated annealing: start "hot" (accept many bad moves to escape local
    minima) and gradually "cool" (become greedy) until the system settles
    into a near-optimal assignment.

Problem Formulation (QUBO / Ising):
    Minimize the energy:
        E = w_balance * imbalance(assignment)
          + w_affinity * affinity_penalty(assignment)

    where `imbalance` is the variance of per-GPU load and
    `affinity_penalty` discourages splitting communicating tasks across
    fault domains.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A schedulable unit of work."""

    task_id: str
    load: float  # Relative compute cost (e.g. FLOPs or samples/step)
    affinity_group: Optional[str] = None  # Tasks that communicate heavily


@dataclass
class SchedulerConfig:
    """Simulated-annealing hyper-parameters."""

    initial_temp: float = 10.0
    final_temp: float = 0.01
    cooling_rate: float = 0.95
    iterations_per_temp: int = 50
    balance_weight: float = 1.0
    affinity_weight: float = 0.5
    seed: Optional[int] = None


@dataclass
class Schedule:
    """The result of a scheduling run: task_id -> gpu_index."""

    assignment: Dict[str, int]
    energy: float
    per_gpu_load: List[float] = field(default_factory=list)


class QuantumInspiredScheduler:
    """
    Assigns tasks to GPUs by minimizing an energy function via simulated
    annealing (a quantum-annealing-inspired metaheuristic).
    """

    def __init__(self, num_gpus: int, config: Optional[SchedulerConfig] = None):
        if num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")
        self.num_gpus = num_gpus
        self.config = config or SchedulerConfig()
        self._rng = random.Random(self.config.seed)

    def _per_gpu_load(self, tasks: List[Task], assignment: Dict[str, int]) -> List[float]:
        loads = [0.0] * self.num_gpus
        by_id = {t.task_id: t for t in tasks}
        for task_id, gpu in assignment.items():
            loads[gpu] += by_id[task_id].load
        return loads

    def _energy(self, tasks: List[Task], assignment: Dict[str, int]) -> float:
        """Compute the energy (lower is better) of an assignment."""
        loads = self._per_gpu_load(tasks, assignment)
        mean = sum(loads) / self.num_gpus
        imbalance = sum((load - mean) ** 2 for load in loads) / self.num_gpus

        # Affinity penalty: communicating tasks on different GPUs cost extra.
        affinity_penalty = 0.0
        groups: Dict[str, List[int]] = {}
        for task in tasks:
            if task.affinity_group is None:
                continue
            groups.setdefault(task.affinity_group, []).append(assignment[task.task_id])
        for gpus in groups.values():
            affinity_penalty += len(set(gpus)) - 1  # 0 if all co-located

        return (
            self.config.balance_weight * imbalance
            + self.config.affinity_weight * affinity_penalty
        )

    def schedule(self, tasks: List[Task]) -> Schedule:
        """
        Run simulated annealing to find a low-energy task->GPU assignment.
        """
        if not tasks:
            return Schedule(assignment={}, energy=0.0, per_gpu_load=[0.0] * self.num_gpus)

        # Initial random assignment.
        current = {t.task_id: self._rng.randrange(self.num_gpus) for t in tasks}
        current_energy = self._energy(tasks, current)
        best = dict(current)
        best_energy = current_energy

        temp = self.config.initial_temp
        while temp > self.config.final_temp:
            for _ in range(self.config.iterations_per_temp):
                # Neighbour: move a random task to a random GPU.
                task = self._rng.choice(tasks)
                old_gpu = current[task.task_id]
                new_gpu = self._rng.randrange(self.num_gpus)
                if new_gpu == old_gpu:
                    continue

                current[task.task_id] = new_gpu
                new_energy = self._energy(tasks, current)
                delta = new_energy - current_energy

                # Metropolis acceptance criterion.
                if delta < 0 or self._rng.random() < math.exp(-delta / temp):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best = dict(current)
                else:
                    current[task.task_id] = old_gpu  # Revert.

            temp *= self.config.cooling_rate

        logger.info(
            "Scheduled %d tasks across %d GPUs (energy=%.4f)",
            len(tasks),
            self.num_gpus,
            best_energy,
        )
        return Schedule(
            assignment=best,
            energy=best_energy,
            per_gpu_load=self._per_gpu_load(tasks, best),
        )
