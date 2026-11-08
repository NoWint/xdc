"""Resource-aware node selection and pipeline layer allocation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .state import NodeRecord, StateStore


def is_online(node: NodeRecord, timeout_s: float) -> bool:
    return (time.time() - node.last_seen) <= timeout_s and not node.banned


def node_score(node: NodeRecord) -> float:
    caps = node.caps()
    memory_factor = min(1.0, max(0.1, caps.available_memory_mb / max(caps.total_memory_mb, 1)))
    return max(caps.compute_score, 0.1) * max(node.reputation, 0.05) * (0.5 + 0.5 * memory_factor)


def group_by_island(nodes: list[NodeRecord]) -> list[NodeRecord]:
    """Order nodes so each island occupies one contiguous stage block.

    A pipeline only pays a cross-island hop at the boundary between two island
    blocks.  Ordering by score alone can interleave islands and multiply WAN
    traffic.  Islands are ordered by their best node score, nodes within an
    island by score.
    """
    buckets: dict[str, list[NodeRecord]] = {}
    for node in nodes:
        buckets.setdefault(node.labels.get("island", "default"), []).append(node)
    ordered: list[NodeRecord] = []
    for island in sorted(
        buckets,
        key=lambda name: max(node_score(n) for n in buckets[name]),
        reverse=True,
    ):
        ordered.extend(sorted(buckets[island], key=node_score, reverse=True))
    return ordered


def training_candidates(store: StateStore, timeout_s: float, max_n: int) -> list[NodeRecord]:
    nodes = [
        n
        for n in store.list_nodes()
        if is_online(n, timeout_s) and not n.banned and n.reputation >= 0.15
    ]
    nodes.sort(key=node_score, reverse=True)
    return nodes[:max_n]


def inference_candidates(store: StateStore, timeout_s: float) -> list[NodeRecord]:
    # Tunnel liveness is checked against the actual hub by the inference engine;
    # the persisted ``tunnel_connected`` flag can lag a reconnect by one event.
    nodes = [
        n
        for n in store.list_nodes()
        if is_online(n, timeout_s) and not n.banned and n.reputation >= 0.15
    ]
    nodes.sort(key=node_score, reverse=True)
    return nodes


@dataclass
class StagePlan:
    node: NodeRecord
    stage_index: int
    layer_start: int
    layer_end: int

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start


def allocate_stages(nodes: list[NodeRecord], n_layers: int, max_stages: int = 0) -> list[StagePlan]:
    """Allocate contiguous layer ranges proportional to measured compute speed.

    Every stage gets at least one layer.  Faster/more reliable nodes receive more
    layers, which is the practical form of heterogeneous pipeline balancing here.
    """
    if not nodes:
        raise ValueError("no inference workers available")
    stages = min(len(nodes), max(1, n_layers))
    if max_stages > 0:
        stages = min(stages, max_stages)
    nodes = nodes[:stages]
    # Layer count should follow compute, not transient free memory.  The latter
    # is dominated by page cache and changes every heartbeat, which produced
    # badly unbalanced 23/7 splits on identical workers.
    # Startup benchmarks on co-located workers are noisy (observed 32 vs 7
    # GFLOPS on identical CPUs).  A log compress keeps a real 4x hardware gap
    # meaningful without producing a 24/6 split on the same machine.
    weights = [
        math.log1p(max(n.caps().compute_score, 0.1) * max(n.reputation, 0.05))
        for n in nodes
    ]
    # Embedding/head work is concentrated at the ends; slightly discount endpoints.
    if stages > 1:
        weights[0] *= 0.9
        weights[-1] *= 0.9
    total = sum(weights)
    raw = [n_layers * w / total for w in weights]
    counts = [max(1, int(x)) for x in raw]
    while sum(counts) > n_layers:
        # remove from the largest count (but never below 1)
        idx = max((i for i in range(stages) if counts[i] > 1), key=lambda i: counts[i], default=None)
        if idx is None:
            break
        counts[idx] -= 1
    while sum(counts) < n_layers:
        fractions = [raw[i] - counts[i] for i in range(stages)]
        counts[max(range(stages), key=lambda i: fractions[i])] += 1
    plans = []
    start = 0
    for i, (node, count) in enumerate(zip(nodes, counts)):
        plans.append(StagePlan(node=node, stage_index=i, layer_start=start, layer_end=start + count))
        start += count
    assert start == n_layers
    return plans


def pooled_resources(nodes: list[NodeRecord], timeout_s: float, store: StateStore) -> dict:
    online = [n for n in nodes if is_online(n, timeout_s)]
    caps = [n.caps() for n in online]
    return {
        "nodes_online": len(online),
        "pooled_cpu_threads": sum(c.cpu_count for c in caps),
        "pooled_memory_mb": sum(c.available_memory_mb for c in caps),
        "pooled_gpu_memory_mb": sum(c.gpu_memory_mb for c in caps),
    }
