"""Stability analysis and truth refinement."""

import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from rbn_train.config import AnalysisConfig
from rbn_train.types import LabeledCluster, StabilityKey, StableBin, FinalCluster, Cluster

logger = logging.getLogger(__name__)


def levenshtein_distance(a: str, b: str) -> int:
    """Compute edit distance between two strings.
    
    Uses dynamic programming with unit costs for insertions, deletions,
    and substitutions. Match operations have zero cost.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        Minimum number of edit operations to transform a into b
    """
    # Handle empty strings
    if not a:
        return len(b)
    if not b:
        return len(a)
    
    # Create DP table
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Cost of deleting all characters from a
    for j in range(n + 1):
        dp[0][j] = j  # Cost of inserting all characters of b
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                # Characters match, no cost
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Take minimum of insert, delete, substitute
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete from a
                    dp[i][j - 1],      # Insert into a
                    dp[i - 1][j - 1],  # Substitute
                )
    
    return dp[m][n]


def build_stability_bins(labeled_clusters: List[LabeledCluster],
                        config: AnalysisConfig) -> Dict[StabilityKey, List[LabeledCluster]]:
    """Group clusters into stability bins.
    
    Clusters are grouped into larger frequency bins for stability validation.
    Only clusters with provisional_true_call are included.
    
    Args:
        labeled_clusters: List of LabeledCluster objects
        config: Analysis configuration with stability binning parameters
        
    Returns:
        Dictionary mapping StabilityKey to list of clusters in that bin
    """
    bins: Dict[StabilityKey, List[LabeledCluster]] = defaultdict(list)
    
    for labeled_cluster in labeled_clusters:
        # Skip clusters without provisional truth
        if labeled_cluster.provisional_true_call is None:
            continue
        
        # Compute center frequency
        spots = labeled_cluster.cluster.spots
        if not spots:
            continue
        
        center_freq_hz = sum(spot.freq_hz for spot in spots) / len(spots)
        
        # Compute stability frequency bin
        freq_bin = int(center_freq_hz) // config.stability_freq_bin_hz
        
        # Create stability key
        key = StabilityKey(
            band=labeled_cluster.cluster.key.band,
            mode=labeled_cluster.cluster.key.mode,
            freq_bin=freq_bin,
        )
        
        # Add to bin
        bins[key].append(labeled_cluster)
    
    return bins


def compute_stable_calls(bins: Dict[StabilityKey, List[LabeledCluster]],
                        config: AnalysisConfig) -> Dict[StabilityKey, StableBin]:
    """Determine stable callsign per bin.
    
    The stable callsign is the one that appears most frequently across
    clusters in the bin, subject to minimum thresholds and tie-breaking.
    
    Args:
        bins: Dictionary mapping StabilityKey to list of clusters
        config: Analysis configuration with stability thresholds
        
    Returns:
        Dictionary mapping StabilityKey to StableBin with stable_call
    """
    stable_bins = {}
    
    for key, clusters in bins.items():
        total_clusters = len(clusters)
        
        # Check minimum clusters threshold
        if total_clusters < config.stability_min_clusters:
            stable_bins[key] = StableBin(key=key, stable_call=None)
            continue
        
        # Count frequency of each provisional_true_call
        call_counts: Dict[str, int] = defaultdict(int)
        for cluster in clusters:
            if cluster.provisional_true_call is not None:
                call_counts[cluster.provisional_true_call] += 1
        
        if not call_counts:
            stable_bins[key] = StableBin(key=key, stable_call=None)
            continue
        
        # Find callsign(s) with most clusters
        max_count = max(call_counts.values())
        best_calls = [call for call, count in call_counts.items() if count == max_count]
        
        # Check for tie
        if len(best_calls) > 1:
            stable_bins[key] = StableBin(key=key, stable_call=None)
            continue
        
        best_call = best_calls[0]
        
        # Check minimum clusters threshold for best call
        if max_count < config.stability_min_clusters:
            stable_bins[key] = StableBin(key=key, stable_call=None)
            continue
        
        # Check minimum share threshold
        share_percent = 100.0 * max_count / total_clusters
        if share_percent < config.stability_min_share_percent:
            stable_bins[key] = StableBin(key=key, stable_call=None)
            continue
        
        # All thresholds met
        stable_bins[key] = StableBin(key=key, stable_call=best_call)
    
    return stable_bins


def refine_truth(labeled_clusters: List[LabeledCluster],
                stable_bins: Dict[StabilityKey, StableBin],
                config: AnalysisConfig) -> List[FinalCluster]:
    """Refine provisional truth using stability analysis.
    
    Combines provisional truth with stability validation and Levenshtein
    distance matching to produce final true callsigns.
    
    Args:
        labeled_clusters: List of LabeledCluster objects
        stable_bins: Dictionary of StableBin objects from stability analysis
        config: Analysis configuration
        
    Returns:
        List of FinalCluster objects with true_call_final assigned
    """
    final_clusters = []
    
    for labeled_cluster in labeled_clusters:
        # If no provisional truth, final truth is also None
        if labeled_cluster.provisional_true_call is None:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=None,
            ))
            continue
        
        # Compute center frequency and stability key
        spots = labeled_cluster.cluster.spots
        if not spots:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=labeled_cluster.provisional_true_call,
            ))
            continue
        
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
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=labeled_cluster.provisional_true_call,
            ))
            continue
        
        stable_call = stable_bin.stable_call
        provisional_call = labeled_cluster.provisional_true_call
        
        # If stable call equals provisional call, use it
        if stable_call == provisional_call:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=provisional_call,
            ))
            continue
        
        # Get all decoded calls in cluster
        decoded_calls = {spot.dx_call for spot in spots}
        
        # If stable call appears in decoded calls, use it
        if stable_call in decoded_calls:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=stable_call,
            ))
            continue
        
        # Check Levenshtein distance to decoded calls
        min_distance = min(
            levenshtein_distance(stable_call, decoded_call)
            for decoded_call in decoded_calls
        )

        # If distance <= 2, use stable call
        if min_distance <= 2:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=stable_call,
            ))
        else:
            # Otherwise, fall back to provisional truth
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=provisional_call,
            ))

    return final_clusters


# Worker function for parallel processing (must be at module level for pickling)
def _refine_cluster_batch(args: Tuple[List[LabeledCluster], dict, dict]) -> List[FinalCluster]:
    """Process a batch of labeled clusters for truth refinement.

    Args:
        args: Tuple of (labeled_clusters, stable_bins_serialized, config_dict)

    Returns:
        List of FinalCluster objects
    """
    labeled_clusters, stable_bins_serialized, config_dict = args
    config = AnalysisConfig(**config_dict)

    # Reconstruct stable_bins from serialized form
    stable_bins = {}
    for key_tuple, stable_call in stable_bins_serialized.items():
        key = StabilityKey(*key_tuple)
        stable_bins[key] = StableBin(key=key, stable_call=stable_call)

    final_clusters = []

    for labeled_cluster in labeled_clusters:
        # If no provisional truth, final truth is also None
        if labeled_cluster.provisional_true_call is None:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=None,
            ))
            continue

        # Compute center frequency and stability key
        spots = labeled_cluster.cluster.spots
        if not spots:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=labeled_cluster.provisional_true_call,
            ))
            continue

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
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=labeled_cluster.provisional_true_call,
            ))
            continue

        stable_call = stable_bin.stable_call
        provisional_call = labeled_cluster.provisional_true_call

        # If stable call equals provisional call, use it
        if stable_call == provisional_call:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=provisional_call,
            ))
            continue

        # Get all decoded calls in cluster
        decoded_calls = {spot.dx_call for spot in spots}

        # If stable call appears in decoded calls, use it
        if stable_call in decoded_calls:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=stable_call,
            ))
            continue

        # Check Levenshtein distance to decoded calls
        min_distance = min(
            levenshtein_distance(stable_call, decoded_call)
            for decoded_call in decoded_calls
        )

        # If distance <= 2, use stable call
        if min_distance <= 2:
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=stable_call,
            ))
        else:
            # Otherwise, fall back to provisional truth
            final_clusters.append(FinalCluster(
                cluster=labeled_cluster.cluster,
                true_call_final=provisional_call,
            ))

    return final_clusters


def refine_truth_parallel(
    labeled_clusters: List[LabeledCluster],
    stable_bins: Dict[StabilityKey, StableBin],
    config: AnalysisConfig,
    workers: int = 1,
    chunk_size: int = 1000
) -> List[FinalCluster]:
    """Refine provisional truth using stability analysis (parallel version).

    Uses multiprocessing to distribute cluster processing across workers.

    Args:
        labeled_clusters: List of LabeledCluster objects
        stable_bins: Dictionary of StableBin objects from stability analysis
        config: Analysis configuration
        workers: Number of worker processes
        chunk_size: Number of clusters per worker batch

    Returns:
        List of FinalCluster objects with true_call_final assigned
    """
    if not labeled_clusters:
        return []

    # Sequential processing for single worker
    if workers <= 1:
        return refine_truth(labeled_clusters, stable_bins, config)

    logger.info(f"Refining truth for {len(labeled_clusters)} clusters with {workers} workers")

    from concurrent.futures import ProcessPoolExecutor

    # Serialize config for passing to workers
    config_dict = asdict(config)

    # Serialize stable_bins: convert StabilityKey to tuple for pickling
    stable_bins_serialized = {}
    for key, stable_bin in stable_bins.items():
        key_tuple = (key.band, key.mode, key.freq_bin)
        stable_bins_serialized[key_tuple] = stable_bin.stable_call

    # Split labeled_clusters into batches
    batches = []
    for i in range(0, len(labeled_clusters), chunk_size):
        batch = labeled_clusters[i:i + chunk_size]
        batches.append((batch, stable_bins_serialized, config_dict))

    logger.debug(f"Split into {len(batches)} batches of up to {chunk_size} clusters each")

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_refine_cluster_batch, batches))

    # Flatten results
    all_final = []
    for batch_result in results:
        all_final.extend(batch_result)

    logger.info(f"Completed parallel refinement of {len(all_final)} clusters")
    return all_final
