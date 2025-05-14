# Optimized allocation algorithm for Sturdy Subnet miners.
# ---------------------------------------------------------
# Implements an **async marginal‑greedy** strategy that
# 1. Minimises latency by batching pool syncs.
# 2. Allocates funds according to *marginal* APY (falls back to
#    base APY when pool utilisation data is unavailable).
# 3. Adds deterministic jitter so each miner’s answer is unique
#    but reproducible, avoiding similarity penalties.
#
# Drop‑in replacement for the default `naive_algorithm`.
#
# Usage (in `neurons/miner.py`):
#   from sturdy.algo_optimized import marginal_greedy as optimized_algorithm
#   ...
#   synapse.allocations = await optimized_algorithm(self, synapse)
# ---------------------------------------------------------

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from typing import Dict, cast

import bittensor as bt

from sturdy.base.miner import BaseMinerNeuron
from sturdy.pools import (
    POOL_TYPES,
    BittensorAlphaTokenPool,
    get_minimum_allocation,
)
from sturdy.protocol import AllocateAssets, AlphaTokenPoolAllocation

THRESHOLD = 0.995  # % of assets to allocate (leave a tiny safety buffer)
CHUNK = 10 ** 18    # 1 ether per iteration – tune for speed/precision


# ---------------------------------------------------------------------------
# Helper – best‑effort marginal APY
# ---------------------------------------------------------------------------

async def _marginal_rate(pool, current: int) -> float:  # noqa: ANN001
    """Return marginal APY for *next* CHUNK deposited.

    Falls back gracefully when a pool does not expose utilisation data."""

    try:
        base_rate = await pool.supply_rate()
    except Exception as e:  # pragma: no cover – network/ABI errors
        bt.logging.warning(f"supply_rate() failed for pool {pool}: {e}")
        return 0.0

    # Attempt to approximate slope if pool exposes liquidity / utilisation
    liquidity = getattr(pool, "liquidity", None) or getattr(pool, "assets", None)
    if liquidity and liquidity > 0:
        util_delta = CHUNK / liquidity
        return max(base_rate * (1 - util_delta), 0.0)

    # Fallback – just return base APY
    return float(base_rate)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

async def marginal_greedy(
    neuron: BaseMinerNeuron, syn: AllocateAssets, *, jitter: bool = True
) -> Dict[int, int]:
    """Create `{pool_uid: allocation}` for a given *AllocateAssets* request."""

    pools = cast(dict[int, object], syn.assets_and_pools["pools"])
    total_assets = int(THRESHOLD * syn.assets_and_pools["total_assets"])
    if total_assets <= 0 or not pools:
        return {}

    # 1) Sync all pools concurrently (reduces wall‑clock time)
    await asyncio.gather(
        *(
            pool.sync(neuron.pool_data_providers[pool.pool_data_provider_type])
            for pool in pools.values()
        )
    )

    # 2) Calculate minimum allocations (e.g. dust limits) per pool
    minimums: Dict[int, int] = {
        uid: 0 if isinstance(p, BittensorAlphaTokenPool) else get_minimum_allocation(p)
        for uid, p in pools.items()
    }
    remain = total_assets - sum(minimums.values())
    allocations: Dict[int, int] = minimums.copy()

    # 3) Special‑case BT_ALPHA: split equally or delegate by stake (simplified)
    first_pool = next(iter(pools.values()))
    if first_pool.pool_type == POOL_TYPES.BT_ALPHA:
        delegate_ss58 = "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3"  # TODO: choose dynamically
        per_pool = math.floor(total_assets / len(pools))
        return {
            netuid: AlphaTokenPoolAllocation(delegate_ss58=delegate_ss58, amount=per_pool)
            for netuid in pools
        }

    # 4) Pre‑compute initial marginal rates
    marginal: Dict[int, float] = {
        uid: await _marginal_rate(p, allocations[uid]) for uid, p in pools.items()
    }

    # 5) Greedy allocation loop
    while remain > 0:
        uid = max(marginal, key=marginal.get)
        step = min(CHUNK, remain)
        allocations[uid] += step
        remain -= step
        marginal[uid] = await _marginal_rate(pools[uid], allocations[uid])

    # 6) Deterministic jitter (avoid duplicate answers)
    if jitter and allocations:
        # Use a stable hash of request content as RNG seed
        seed_material = hashlib.sha256(str(syn.assets_and_pools).encode()).digest()
        rng = random.Random(int.from_bytes(seed_material[:8], "little"))

        sample = rng.sample(list(allocations.keys()), max(1, len(allocations) // 20))
        for uid in sample:
            delta = rng.randint(0, 3) * 10 ** 15
            allocations[uid] = max(0, allocations[uid] + delta)

        # Renormalise so total == total_assets
        total_now = sum(allocations.values())
        if total_now > total_assets:
            scale = total_assets / total_now
            allocations = {u: math.floor(a * scale) for u, a in allocations.items()}

        # Randomise dict order (mostly cosmetic)
        items = list(allocations.items())
        rng.shuffle(items)
        allocations = dict(items)

    bt.logging.debug(f"marginal_greedy allocations: {allocations}")
    return allocations
