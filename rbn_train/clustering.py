"""Micro-clustering and provisional truth determination."""

import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from rbn_train.config import AnalysisConfig
from rbn_train.types import Spot, ClusterKey, Cluster, LabeledCluster

logger = logging.getLogger(__name__)


def build_micro_clusters(spots: List[Spot], config: AnalysisConfig) -> List[Cluster]:
    """Group spots into micro-clusters by time and frequency.
    
    Spots are grouped into micro-clusters based on band, mode, frequency bin,
    and time bin. This identifies observations of the same signal from different
    skimmers within a small time and frequency window.
    
    Args:
        spots: List of Spot objects to cluster
        config: Analysis configuration with binning parameters
        
    Returns:
        List of Cluster objects, each containing spots with the same ClusterKey
    """
    # Group spots by ClusterKey
    clusters_dict: Dict[ClusterKey, List[Spot]] = defaultdict(list)
    
    for spot in spots:
        # Compute frequency bin
        freq_bin = int(spot.freq_hz) // config.cluster_freq_bin_hz
        
        # Compute time bin
        time_bin = int(spot.timestamp.timestamp()) // config.cluster_time_seconds
        
        # Create cluster key
        key = ClusterKey(
            band=spot.band,
            mode=spot.mode,
            freq_bin=freq_bin,
            time_bin=time_bin,
        )
        
        # Add spot to cluster
        clusters_dict[key].append(spot)
    
    # Convert to list of Cluster objects
    clusters = [
        Cluster(key=key, spots=spots_list)
        for key, spots_list in clusters_dict.items()
    ]
    
    return clusters


def label_provisional_truth(clusters: List[Cluster], config: AnalysisConfig) -> List[LabeledCluster]:
    """Assign provisional true callsign to each cluster.
    
    The provisional true callsign is determined by skimmer consensus within
    the micro-cluster. The callsign with the most distinct skimmers is selected,
    subject to minimum thresholds and tie-breaking rules.
    
    Args:
        clusters: List of Cluster objects
        config: Analysis configuration with threshold parameters
        
    Returns:
        List of LabeledCluster objects with provisional_true_call assigned
    """
    labeled_clusters = []
    
    for cluster in clusters:
        provisional_call = compute_skimmer_consensus(cluster, config)
        labeled_clusters.append(
            LabeledCluster(
                cluster=cluster,
                provisional_true_call=provisional_call,
            )
        )
    
    return labeled_clusters


def compute_skimmer_consensus(cluster: Cluster, config: AnalysisConfig) -> Optional[str]:
    """Determine best callsign by distinct skimmer count.
    
    Counts the number of distinct skimmers reporting each callsign and selects
    the callsign with the most skimmers, subject to thresholds and tie-breaking.
    
    Args:
        cluster: Cluster to analyze
        config: Analysis configuration with threshold parameters
        
    Returns:
        Best callsign if thresholds are met and no tie, None otherwise
    """
    # Build mapping: callsign -> set of skimmers
    call_skimmers: Dict[str, set] = defaultdict(set)
    
    for spot in cluster.spots:
        call_skimmers[spot.dx_call].add(spot.skimmer)
    
    # Compute total unique skimmers
    all_skimmers = set()
    for skimmers in call_skimmers.values():
        all_skimmers.update(skimmers)
    
    total_skimmers = len(all_skimmers)
    
    # If no skimmers, return None
    if total_skimmers == 0:
        return None
    
    # Find callsign(s) with most distinct skimmers
    max_skimmers = 0
    best_calls = []
    
    for call, skimmers in call_skimmers.items():
        num_skimmers = len(skimmers)
        if num_skimmers > max_skimmers:
            max_skimmers = num_skimmers
            best_calls = [call]
        elif num_skimmers == max_skimmers:
            best_calls.append(call)
    
    # Check for tie
    if len(best_calls) > 1:
        return None
    
    best_call = best_calls[0]
    
    # Check minimum skimmers threshold
    if max_skimmers < config.min_cluster_skimmers:
        return None
    
    # Check minimum share threshold
    share_percent = 100.0 * max_skimmers / total_skimmers
    if share_percent < config.min_cluster_share_percent:
        return None

    return best_call


# Worker function for parallel processing (must be at module level for pickling)
def _process_cluster_batch(args: Tuple[List[Cluster], dict]) -> List[LabeledCluster]:
    """Process a batch of clusters for provisional truth.

    Args:
        args: Tuple of (clusters, config_dict)

    Returns:
        List of LabeledCluster objects
    """
    clusters, config_dict = args
    config = AnalysisConfig(**config_dict)

    labeled = []
    for cluster in clusters:
        provisional_call = compute_skimmer_consensus(cluster, config)
        labeled.append(LabeledCluster(
            cluster=cluster,
            provisional_true_call=provisional_call,
        ))
    return labeled


def label_provisional_truth_parallel(
    clusters: List[Cluster],
    config: AnalysisConfig,
    workers: int = 1,
    chunk_size: int = 1000
) -> List[LabeledCluster]:
    """Assign provisional true callsign to each cluster (parallel version).

    Uses multiprocessing to distribute cluster processing across workers.

    Args:
        clusters: List of Cluster objects
        config: Analysis configuration with threshold parameters
        workers: Number of worker processes
        chunk_size: Number of clusters per worker batch

    Returns:
        List of LabeledCluster objects with provisional_true_call assigned
    """
    if not clusters:
        return []

    # Sequential processing for single worker
    if workers <= 1:
        return label_provisional_truth(clusters, config)

    logger.info(f"Processing {len(clusters)} clusters with {workers} workers")

    from concurrent.futures import ProcessPoolExecutor

    # Serialize config for passing to workers
    config_dict = asdict(config)

    # Split clusters into batches
    batches = []
    for i in range(0, len(clusters), chunk_size):
        batch = clusters[i:i + chunk_size]
        batches.append((batch, config_dict))

    logger.debug(f"Split into {len(batches)} batches of up to {chunk_size} clusters each")

    # Process batches in parallel
    all_labeled = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_process_cluster_batch, batches))

    # Flatten results
    for batch_result in results:
        all_labeled.extend(batch_result)

    logger.info(f"Completed parallel processing of {len(all_labeled)} clusters")
    return all_labeled
