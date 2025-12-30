"""Spotter reliability analysis and export.

This module computes per-skimmer accuracy metrics by comparing decoded callsigns
against inferred true callsigns from the analysis pipeline. It provides:

1. Global spotter reliability - overall accuracy per skimmer
2. Band-specific reliability - accuracy broken down by amateur band
3. Mode-specific reliability - accuracy broken down by mode (CW/RTTY)

The reliability data can be used by runtime correction engines to weight
spotter contributions based on historical accuracy.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

from rbn_train.types import FinalCluster


logger = logging.getLogger(__name__)


@dataclass
class SpotterStats:
    """Statistics for a single skimmer station.

    Attributes:
        total: Total number of spots from this skimmer
        exact_matches: Spots where decoded call exactly matched true call
        close_matches: Spots within edit distance 1 of true call (excluding exact)
        busted: Spots that were incorrect (distance > 1)
    """
    total: int = 0
    exact_matches: int = 0
    close_matches: int = 0
    busted: int = 0

    @property
    def reliability(self) -> float:
        """Compute reliability as exact_matches / total."""
        if self.total == 0:
            return 0.0
        return self.exact_matches / self.total

    @property
    def close_reliability(self) -> float:
        """Compute reliability including close matches."""
        if self.total == 0:
            return 0.0
        return (self.exact_matches + self.close_matches) / self.total


@dataclass
class SpotterReliability:
    """Aggregated spotter reliability data.

    Attributes:
        global_stats: Per-skimmer statistics (global, all bands/modes)
        by_band: Per-skimmer statistics broken down by band
        by_mode: Per-skimmer statistics broken down by mode
        min_spots_threshold: Minimum spots required for reliable statistics
    """
    global_stats: Dict[str, SpotterStats] = field(default_factory=dict)
    by_band: Dict[str, Dict[str, SpotterStats]] = field(default_factory=dict)
    by_mode: Dict[str, Dict[str, SpotterStats]] = field(default_factory=dict)
    min_spots_threshold: int = 100


def levenshtein_distance_fast(a: str, b: str, max_dist: int = 2) -> int:
    """Compute Levenshtein distance with early termination.

    Optimized for checking if distance is within a threshold.
    Returns actual distance if <= max_dist, otherwise returns max_dist + 1.

    Args:
        a: First string
        b: Second string
        max_dist: Maximum distance to compute accurately

    Returns:
        Edit distance if <= max_dist, else max_dist + 1
    """
    if a == b:
        return 0

    len_a, len_b = len(a), len(b)

    # Quick rejection based on length difference
    if abs(len_a - len_b) > max_dist:
        return max_dist + 1

    # Ensure a is the shorter string for efficiency
    if len_a > len_b:
        a, b = b, a
        len_a, len_b = len_b, len_a

    # Use single row DP for space efficiency
    prev_row = list(range(len_a + 1))

    for i in range(1, len_b + 1):
        curr_row = [i] + [0] * len_a
        min_in_row = i

        for j in range(1, len_a + 1):
            if a[j - 1] == b[i - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(prev_row[j], curr_row[j - 1], prev_row[j - 1])

            min_in_row = min(min_in_row, curr_row[j])

        # Early termination if minimum exceeds threshold
        if min_in_row > max_dist:
            return max_dist + 1

        prev_row = curr_row

    return prev_row[len_a]


def compute_spotter_reliability(
    final_clusters: List[FinalCluster],
    min_spots: int = 100,
    compute_by_band: bool = True,
    compute_by_mode: bool = True
) -> SpotterReliability:
    """Compute spotter reliability statistics from final clusters.

    For each skimmer, compares decoded callsigns against inferred true callsigns
    to compute accuracy metrics.

    Args:
        final_clusters: List of FinalCluster objects with true_call_final
        min_spots: Minimum spots required for a skimmer to be included
        compute_by_band: Whether to compute band-specific statistics
        compute_by_mode: Whether to compute mode-specific statistics

    Returns:
        SpotterReliability object with computed statistics
    """
    # Initialize statistics containers
    global_stats: Dict[str, SpotterStats] = {}
    by_band: Dict[str, Dict[str, SpotterStats]] = {}  # band -> skimmer -> stats
    by_mode: Dict[str, Dict[str, SpotterStats]] = {}  # mode -> skimmer -> stats

    clusters_processed = 0
    spots_processed = 0

    for fc in final_clusters:
        # Skip clusters without final truth
        if fc.true_call_final is None:
            continue

        clusters_processed += 1
        true_call = fc.true_call_final

        for spot in fc.cluster.spots:
            spots_processed += 1
            skimmer = spot.skimmer
            decoded = spot.dx_call
            band = spot.band
            mode = spot.mode

            # Compute distance
            if decoded == true_call:
                dist = 0
            else:
                dist = levenshtein_distance_fast(decoded, true_call, max_dist=2)

            # Update global stats
            if skimmer not in global_stats:
                global_stats[skimmer] = SpotterStats()

            stats = global_stats[skimmer]
            stats.total += 1
            if dist == 0:
                stats.exact_matches += 1
            elif dist == 1:
                stats.close_matches += 1
            else:
                stats.busted += 1

            # Update band-specific stats
            if compute_by_band:
                if band not in by_band:
                    by_band[band] = {}
                if skimmer not in by_band[band]:
                    by_band[band][skimmer] = SpotterStats()

                band_stats = by_band[band][skimmer]
                band_stats.total += 1
                if dist == 0:
                    band_stats.exact_matches += 1
                elif dist == 1:
                    band_stats.close_matches += 1
                else:
                    band_stats.busted += 1

            # Update mode-specific stats
            if compute_by_mode:
                if mode not in by_mode:
                    by_mode[mode] = {}
                if skimmer not in by_mode[mode]:
                    by_mode[mode][skimmer] = SpotterStats()

                mode_stats = by_mode[mode][skimmer]
                mode_stats.total += 1
                if dist == 0:
                    mode_stats.exact_matches += 1
                elif dist == 1:
                    mode_stats.close_matches += 1
                else:
                    mode_stats.busted += 1

    logger.info(f"Processed {spots_processed} spots from {clusters_processed} clusters")
    logger.info(f"Found {len(global_stats)} unique skimmers")

    return SpotterReliability(
        global_stats=global_stats,
        by_band=by_band,
        by_mode=by_mode,
        min_spots_threshold=min_spots
    )


def export_spotter_reliability(
    reliability: SpotterReliability,
    output_path: Path,
    include_by_band: bool = True,
    include_by_mode: bool = True
) -> Tuple[int, int]:
    """Export spotter reliability to JSON file.

    Only includes skimmers meeting the minimum spots threshold.

    Args:
        reliability: SpotterReliability object
        output_path: Path to write JSON file
        include_by_band: Whether to include band breakdown
        include_by_mode: Whether to include mode breakdown

    Returns:
        Tuple of (skimmers_exported, skimmers_filtered)
    """
    min_spots = reliability.min_spots_threshold

    # Build global stats (filtered)
    skimmers_data = {}
    skimmers_filtered = 0

    for skimmer, stats in reliability.global_stats.items():
        if stats.total < min_spots:
            skimmers_filtered += 1
            continue

        skimmers_data[skimmer] = {
            "total": stats.total,
            "exact": stats.exact_matches,
            "close": stats.close_matches,
            "busted": stats.busted,
            "reliability": round(stats.reliability, 4),
            "close_reliability": round(stats.close_reliability, 4)
        }

    # Build output structure
    data = {
        "min_spots_threshold": min_spots,
        "skimmers_total": len(reliability.global_stats),
        "skimmers_included": len(skimmers_data),
        "skimmers_filtered": skimmers_filtered,
        "skimmers": skimmers_data
    }

    # Add band breakdown if requested
    if include_by_band and reliability.by_band:
        by_band_data = {}
        for band, band_skimmers in reliability.by_band.items():
            by_band_data[band] = {}
            for skimmer, stats in band_skimmers.items():
                # Use lower threshold for band-specific (min_spots / 2)
                band_threshold = max(10, min_spots // 2)
                if stats.total < band_threshold:
                    continue
                by_band_data[band][skimmer] = {
                    "total": stats.total,
                    "exact": stats.exact_matches,
                    "reliability": round(stats.reliability, 4)
                }
        data["by_band"] = by_band_data

    # Add mode breakdown if requested
    if include_by_mode and reliability.by_mode:
        by_mode_data = {}
        for mode, mode_skimmers in reliability.by_mode.items():
            by_mode_data[mode] = {}
            for skimmer, stats in mode_skimmers.items():
                mode_threshold = max(10, min_spots // 2)
                if stats.total < mode_threshold:
                    continue
                by_mode_data[mode][skimmer] = {
                    "total": stats.total,
                    "exact": stats.exact_matches,
                    "reliability": round(stats.reliability, 4)
                }
        data["by_mode"] = by_mode_data

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported {len(skimmers_data)} skimmers to {output_path}")
    logger.info(f"Filtered {skimmers_filtered} skimmers below {min_spots} spots threshold")

    return len(skimmers_data), skimmers_filtered


def export_spotter_reliability_simple(
    reliability: SpotterReliability,
    output_path: Path
) -> int:
    """Export simple spotter reliability file for Go integration.

    Outputs a text file in the format expected by Go's LoadSpotterReliability:
    SKIMMER RELIABILITY

    Args:
        reliability: SpotterReliability object
        output_path: Path to write text file

    Returns:
        Number of skimmers exported
    """
    min_spots = reliability.min_spots_threshold
    lines = []

    # Sort by reliability descending for readability
    sorted_skimmers = sorted(
        reliability.global_stats.items(),
        key=lambda x: x[1].reliability,
        reverse=True
    )

    for skimmer, stats in sorted_skimmers:
        if stats.total < min_spots:
            continue

        rel = round(stats.reliability, 4)
        lines.append(f"{skimmer} {rel}")

    # Write text file
    with open(output_path, 'w') as f:
        f.write(f"# Spotter reliability weights\n")
        f.write(f"# Generated from RBN analysis ({len(reliability.global_stats)} total skimmers)\n")
        f.write(f"# Filtered to skimmers with >= {min_spots} spots\n")
        f.write(f"# Format: SKIMMER RELIABILITY\n")
        f.write("\n".join(lines))
        f.write("\n")

    logger.info(f"Exported {len(lines)} skimmers to {output_path} (simple format)")

    return len(lines)


def export_spotter_reliability_by_mode(
    reliability: SpotterReliability,
    output_dir: Path
) -> Dict[str, int]:
    """Export mode-specific spotter reliability text files.

    Creates separate text files for each mode (e.g., spotter_reliability_cw.txt,
    spotter_reliability_rtty.txt) containing only skimmers with sufficient spots
    in that specific mode.

    Args:
        reliability: SpotterReliability object with by_mode data
        output_dir: Directory to write text files

    Returns:
        Dictionary mapping mode to number of skimmers exported
    """
    min_spots = reliability.min_spots_threshold
    results: Dict[str, int] = {}

    for mode, skimmers in reliability.by_mode.items():
        # Sort by reliability descending
        sorted_skimmers = sorted(
            skimmers.items(),
            key=lambda x: x[1].reliability,
            reverse=True
        )

        lines = []
        for skimmer, stats in sorted_skimmers:
            if stats.total < min_spots:
                continue
            rel = round(stats.reliability, 4)
            lines.append(f"{skimmer} {rel}")

        # Write mode-specific file
        output_path = output_dir / f"spotter_reliability_{mode.lower()}.txt"
        with open(output_path, 'w') as f:
            f.write(f"# Spotter reliability weights for {mode}\n")
            f.write(f"# Generated from RBN analysis\n")
            f.write(f"# Filtered to skimmers with >= {min_spots} {mode} spots\n")
            f.write(f"# Format: SKIMMER RELIABILITY\n")
            f.write("\n".join(lines))
            f.write("\n")

        results[mode] = len(lines)
        logger.info(f"Exported {len(lines)} skimmers to {output_path} ({mode})")

    return results


# Worker function for parallel processing (must be at module level for pickling)
def _compute_reliability_batch(args: Tuple[List[FinalCluster], dict]) -> Tuple[
    Dict[str, Dict[str, int]],  # global_stats
    Dict[str, Dict[str, Dict[str, int]]],  # by_band
    Dict[str, Dict[str, Dict[str, int]]],  # by_mode
]:
    """Process a batch of final clusters for reliability computation.

    Args:
        args: Tuple of (final_clusters, params_dict)

    Returns:
        Tuple of (global_stats_dict, by_band_dict, by_mode_dict)
        Each stats dict has format: {skimmer: {total:, exact:, close:, busted:}}
    """
    final_clusters, params = args
    compute_by_band = params.get('compute_by_band', True)
    compute_by_mode = params.get('compute_by_mode', True)

    # Use simple dicts for pickling compatibility
    global_stats: Dict[str, Dict[str, int]] = {}
    by_band: Dict[str, Dict[str, Dict[str, int]]] = {}
    by_mode: Dict[str, Dict[str, Dict[str, int]]] = {}

    def get_or_create_stats(container: dict, key: str) -> Dict[str, int]:
        if key not in container:
            container[key] = {'total': 0, 'exact': 0, 'close': 0, 'busted': 0}
        return container[key]

    for fc in final_clusters:
        if fc.true_call_final is None:
            continue

        true_call = fc.true_call_final

        for spot in fc.cluster.spots:
            skimmer = spot.skimmer
            decoded = spot.dx_call
            band = spot.band
            mode = spot.mode

            # Compute distance
            if decoded == true_call:
                dist = 0
            else:
                dist = levenshtein_distance_fast(decoded, true_call, max_dist=2)

            # Update global stats
            stats = get_or_create_stats(global_stats, skimmer)
            stats['total'] += 1
            if dist == 0:
                stats['exact'] += 1
            elif dist == 1:
                stats['close'] += 1
            else:
                stats['busted'] += 1

            # Update band-specific stats
            if compute_by_band:
                if band not in by_band:
                    by_band[band] = {}
                band_stats = get_or_create_stats(by_band[band], skimmer)
                band_stats['total'] += 1
                if dist == 0:
                    band_stats['exact'] += 1
                elif dist == 1:
                    band_stats['close'] += 1
                else:
                    band_stats['busted'] += 1

            # Update mode-specific stats
            if compute_by_mode:
                if mode not in by_mode:
                    by_mode[mode] = {}
                mode_stats = get_or_create_stats(by_mode[mode], skimmer)
                mode_stats['total'] += 1
                if dist == 0:
                    mode_stats['exact'] += 1
                elif dist == 1:
                    mode_stats['close'] += 1
                else:
                    mode_stats['busted'] += 1

    return global_stats, by_band, by_mode


def _merge_stats(stats1: Dict[str, int], stats2: Dict[str, int]) -> Dict[str, int]:
    """Merge two stats dictionaries by summing values."""
    return {
        'total': stats1.get('total', 0) + stats2.get('total', 0),
        'exact': stats1.get('exact', 0) + stats2.get('exact', 0),
        'close': stats1.get('close', 0) + stats2.get('close', 0),
        'busted': stats1.get('busted', 0) + stats2.get('busted', 0),
    }


def compute_spotter_reliability_parallel(
    final_clusters: List[FinalCluster],
    min_spots: int = 100,
    compute_by_band: bool = True,
    compute_by_mode: bool = True,
    workers: int = 1,
    chunk_size: int = 1000
) -> SpotterReliability:
    """Compute spotter reliability statistics (parallel version).

    Uses multiprocessing to distribute cluster processing across workers.

    Args:
        final_clusters: List of FinalCluster objects with true_call_final
        min_spots: Minimum spots required for a skimmer to be included
        compute_by_band: Whether to compute band-specific statistics
        compute_by_mode: Whether to compute mode-specific statistics
        workers: Number of worker processes
        chunk_size: Number of clusters per worker batch

    Returns:
        SpotterReliability object with computed statistics
    """
    # Sequential processing for single worker or empty input
    if workers <= 1 or not final_clusters:
        return compute_spotter_reliability(
            final_clusters, min_spots, compute_by_band, compute_by_mode
        )

    logger.info(f"Computing reliability for {len(final_clusters)} clusters with {workers} workers")

    from concurrent.futures import ProcessPoolExecutor

    # Prepare parameters for workers
    params = {
        'compute_by_band': compute_by_band,
        'compute_by_mode': compute_by_mode,
    }

    # Split final_clusters into batches
    batches = []
    for i in range(0, len(final_clusters), chunk_size):
        batch = final_clusters[i:i + chunk_size]
        batches.append((batch, params))

    logger.debug(f"Split into {len(batches)} batches of up to {chunk_size} clusters each")

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_compute_reliability_batch, batches))

    # Merge results
    merged_global: Dict[str, Dict[str, int]] = {}
    merged_by_band: Dict[str, Dict[str, Dict[str, int]]] = {}
    merged_by_mode: Dict[str, Dict[str, Dict[str, int]]] = {}

    for global_stats, by_band, by_mode in results:
        # Merge global stats
        for skimmer, stats in global_stats.items():
            if skimmer not in merged_global:
                merged_global[skimmer] = {'total': 0, 'exact': 0, 'close': 0, 'busted': 0}
            merged_global[skimmer] = _merge_stats(merged_global[skimmer], stats)

        # Merge band stats
        for band, band_skimmers in by_band.items():
            if band not in merged_by_band:
                merged_by_band[band] = {}
            for skimmer, stats in band_skimmers.items():
                if skimmer not in merged_by_band[band]:
                    merged_by_band[band][skimmer] = {'total': 0, 'exact': 0, 'close': 0, 'busted': 0}
                merged_by_band[band][skimmer] = _merge_stats(merged_by_band[band][skimmer], stats)

        # Merge mode stats
        for mode, mode_skimmers in by_mode.items():
            if mode not in merged_by_mode:
                merged_by_mode[mode] = {}
            for skimmer, stats in mode_skimmers.items():
                if skimmer not in merged_by_mode[mode]:
                    merged_by_mode[mode][skimmer] = {'total': 0, 'exact': 0, 'close': 0, 'busted': 0}
                merged_by_mode[mode][skimmer] = _merge_stats(merged_by_mode[mode][skimmer], stats)

    # Convert to SpotterStats objects
    final_global: Dict[str, SpotterStats] = {}
    for skimmer, stats in merged_global.items():
        final_global[skimmer] = SpotterStats(
            total=stats['total'],
            exact_matches=stats['exact'],
            close_matches=stats['close'],
            busted=stats['busted']
        )

    final_by_band: Dict[str, Dict[str, SpotterStats]] = {}
    for band, band_skimmers in merged_by_band.items():
        final_by_band[band] = {}
        for skimmer, stats in band_skimmers.items():
            final_by_band[band][skimmer] = SpotterStats(
                total=stats['total'],
                exact_matches=stats['exact'],
                close_matches=stats['close'],
                busted=stats['busted']
            )

    final_by_mode: Dict[str, Dict[str, SpotterStats]] = {}
    for mode, mode_skimmers in merged_by_mode.items():
        final_by_mode[mode] = {}
        for skimmer, stats in mode_skimmers.items():
            final_by_mode[mode][skimmer] = SpotterStats(
                total=stats['total'],
                exact_matches=stats['exact'],
                close_matches=stats['close'],
                busted=stats['busted']
            )

    logger.info(f"Completed parallel reliability computation for {len(final_global)} skimmers")

    return SpotterReliability(
        global_stats=final_global,
        by_band=final_by_band,
        by_mode=final_by_mode,
        min_spots_threshold=min_spots
    )
