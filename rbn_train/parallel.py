"""Parallel processing utilities for RBN analysis.

This module provides infrastructure for parallelizing CPU-bound operations
in the RBN analysis pipeline using multiprocessing.

Key features:
- Worker pool management with configurable worker count
- Chunked processing for memory efficiency
- Progress logging for long-running parallel operations
- Graceful fallback to sequential processing when workers=1

Usage:
    from rbn_train.parallel import parallel_map, get_worker_count

    # Process items in parallel
    results = parallel_map(process_func, items, workers=4, chunk_size=100)
"""

import os
import logging
from typing import TypeVar, Callable, List, Iterable, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


def get_worker_count(requested: Optional[int] = None) -> int:
    """Determine the number of worker processes to use.

    Args:
        requested: Requested number of workers (None for auto-detect)

    Returns:
        Number of workers to use (minimum 1)
    """
    if requested is not None and requested > 0:
        return requested

    # Auto-detect: use CPU count minus 1, with minimum of 1
    try:
        cpus = cpu_count()
        return max(1, cpus - 1)
    except Exception:
        return 1


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into chunks of specified size.

    Args:
        items: List to split
        chunk_size: Maximum size of each chunk

    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        return [items]

    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    workers: int = 1,
    desc: str = "Processing"
) -> List[R]:
    """Apply a function to items in parallel.

    Args:
        func: Function to apply to each item
        items: List of items to process
        workers: Number of worker processes
        desc: Description for logging

    Returns:
        List of results in same order as input
    """
    if not items:
        return []

    # Sequential processing for single worker
    if workers <= 1:
        logger.debug(f"{desc}: processing {len(items)} items sequentially")
        return [func(item) for item in items]

    logger.info(f"{desc}: processing {len(items)} items with {workers} workers")

    # Parallel processing
    results = [None] * len(items)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks with their indices
        future_to_idx = {
            executor.submit(func, item): idx
            for idx, item in enumerate(items)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(f"{desc}: error processing item {idx}: {e}")
                raise

            completed += 1
            if completed % 1000 == 0:
                logger.debug(f"{desc}: {completed}/{len(items)} items completed")

    logger.debug(f"{desc}: completed all {len(items)} items")
    return results


def parallel_map_chunks(
    func: Callable[[List[T]], List[R]],
    items: List[T],
    workers: int = 1,
    chunk_size: int = 1000,
    desc: str = "Processing"
) -> List[R]:
    """Apply a function to chunks of items in parallel.

    More efficient than parallel_map for large item counts because
    it reduces serialization overhead by batching items.

    Args:
        func: Function that processes a list of items and returns list of results
        items: List of items to process
        workers: Number of worker processes
        chunk_size: Number of items per chunk
        desc: Description for logging

    Returns:
        List of results in same order as input
    """
    if not items:
        return []

    # Sequential processing for single worker
    if workers <= 1:
        logger.debug(f"{desc}: processing {len(items)} items sequentially")
        return func(items)

    # Split into chunks
    chunks = chunk_list(items, chunk_size)
    logger.info(f"{desc}: processing {len(items)} items in {len(chunks)} chunks with {workers} workers")

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all chunks
        future_to_idx = {
            executor.submit(func, chunk): idx
            for idx, chunk in enumerate(chunks)
        }

        # Collect results in order
        chunk_results = [None] * len(chunks)
        completed = 0

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                chunk_results[idx] = future.result()
            except Exception as e:
                logger.error(f"{desc}: error processing chunk {idx}: {e}")
                raise

            completed += 1
            if completed % 10 == 0 or completed == len(chunks):
                logger.debug(f"{desc}: {completed}/{len(chunks)} chunks completed")

    # Flatten results
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)

    logger.debug(f"{desc}: completed all {len(items)} items")
    return results


# Worker functions for parallel processing
# These must be defined at module level for pickling

def _process_cluster_for_truth(args: tuple) -> tuple:
    """Process a single cluster for provisional truth determination.

    Args:
        args: Tuple of (cluster, config_dict)

    Returns:
        Tuple of (cluster, provisional_true_call)
    """
    from rbn_train.clustering import compute_skimmer_consensus
    from rbn_train.config import AnalysisConfig

    cluster, config_dict = args
    config = AnalysisConfig(**config_dict)
    provisional_call = compute_skimmer_consensus(cluster, config)
    return (cluster, provisional_call)


def _process_cluster_for_refinement(args: tuple) -> tuple:
    """Process a single labeled cluster for truth refinement.

    Args:
        args: Tuple of (labeled_cluster, stable_bins_dict, config_dict)

    Returns:
        Tuple of (cluster, true_call_final)
    """
    from rbn_train.types import StabilityKey, StableBin
    from rbn_train.stability import levenshtein_distance
    from rbn_train.config import AnalysisConfig

    labeled_cluster, stable_bins_serialized, config_dict = args
    config = AnalysisConfig(**config_dict)

    # Reconstruct stable_bins from serialized form
    stable_bins = {}
    for key_tuple, bin_data in stable_bins_serialized.items():
        key = StabilityKey(*key_tuple)
        stable_bins[key] = StableBin(key=key, stable_call=bin_data)

    # If no provisional truth, final truth is also None
    if labeled_cluster.provisional_true_call is None:
        return (labeled_cluster.cluster, None)

    # Compute center frequency and stability key
    spots = labeled_cluster.cluster.spots
    if not spots:
        return (labeled_cluster.cluster, labeled_cluster.provisional_true_call)

    center_freq_hz = sum(spot.freq_hz for spot in spots) / len(spots)
    freq_bin = int(center_freq_hz) // config.stability_freq_bin_hz

    stability_key = StabilityKey(
        band=labeled_cluster.cluster.key.band,
        mode=labeled_cluster.cluster.key.mode,
        freq_bin=freq_bin,
    )

    # Look up stability bin
    stable_bin = stable_bins.get(stability_key)

    # If no stability bin or no stable call, use provisional truth
    if stable_bin is None or stable_bin.stable_call is None:
        return (labeled_cluster.cluster, labeled_cluster.provisional_true_call)

    stable_call = stable_bin.stable_call
    provisional_call = labeled_cluster.provisional_true_call

    # If stable call equals provisional call, use it
    if stable_call == provisional_call:
        return (labeled_cluster.cluster, provisional_call)

    # Get all decoded calls in cluster
    decoded_calls = {spot.dx_call for spot in spots}

    # If stable call appears in decoded calls, use it
    if stable_call in decoded_calls:
        return (labeled_cluster.cluster, stable_call)

    # Check Levenshtein distance to decoded calls
    min_distance = min(
        levenshtein_distance(stable_call, decoded_call)
        for decoded_call in decoded_calls
    )

    # If distance <= 2, use stable call
    if min_distance <= 2:
        return (labeled_cluster.cluster, stable_call)
    else:
        return (labeled_cluster.cluster, provisional_call)
