"""Band-specific and segment priors generation and export.

This module extends the basic global priors with:

1. Band priors - callsign frequency counts per amateur band
2. Segment priors - callsign frequency counts per frequency segment within bands

These provide more granular prior knowledge for runtime correction, recognizing
that callsign populations differ significantly between bands and operating modes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

from rbn_train.types import FinalCluster
from rbn_train.confusion import Alphabet, normalize_call


logger = logging.getLogger(__name__)


# Default segment definitions for HF bands
# Each segment is a tuple of (start_khz, end_khz, name)
DEFAULT_SEGMENTS: Dict[str, List[Tuple[float, float, str]]] = {
    "160m": [
        (1800, 1840, "160m_cw"),
        (1840, 2000, "160m_other"),
    ],
    "80m": [
        (3500, 3600, "80m_cw"),
        (3600, 3800, "80m_other"),
    ],
    "40m": [
        (7000, 7050, "40m_cw"),
        (7050, 7300, "40m_other"),
    ],
    "30m": [
        (10100, 10150, "30m_all"),
    ],
    "20m": [
        (14000, 14070, "20m_cw"),
        (14070, 14100, "20m_rtty"),
        (14100, 14350, "20m_other"),
    ],
    "17m": [
        (18068, 18110, "17m_cw"),
        (18110, 18168, "17m_other"),
    ],
    "15m": [
        (21000, 21070, "15m_cw"),
        (21070, 21150, "15m_rtty"),
        (21150, 21450, "15m_other"),
    ],
    "12m": [
        (24890, 24930, "12m_cw"),
        (24930, 24990, "12m_other"),
    ],
    "10m": [
        (28000, 28070, "10m_cw"),
        (28070, 28150, "10m_rtty"),
        (28150, 29700, "10m_other"),
    ],
    "6m": [
        (50000, 50100, "6m_cw"),
        (50100, 54000, "6m_other"),
    ],
}


@dataclass
class BandPriors:
    """Band-specific callsign frequency counts.

    Attributes:
        global_counts: Global callsign counts (all bands)
        by_band: Per-band callsign counts
        bands_seen: List of bands that have data
    """
    global_counts: Dict[str, int] = field(default_factory=dict)
    by_band: Dict[str, Dict[str, int]] = field(default_factory=dict)
    bands_seen: List[str] = field(default_factory=list)


@dataclass
class SegmentPriors:
    """Frequency segment callsign counts.

    Attributes:
        by_segment: Per-segment callsign counts
        segment_info: Segment metadata (frequency range, band)
    """
    by_segment: Dict[str, Dict[str, int]] = field(default_factory=dict)
    segment_info: Dict[str, Dict] = field(default_factory=dict)


def compute_band_priors(
    final_clusters: List[FinalCluster],
    alphabet: Optional[Alphabet] = None
) -> BandPriors:
    """Compute band-specific callsign priors from final clusters.

    Args:
        final_clusters: List of FinalCluster objects with true_call_final
        alphabet: Optional alphabet for callsign normalization

    Returns:
        BandPriors object with global and per-band counts
    """
    global_counts: Dict[str, int] = {}
    by_band: Dict[str, Dict[str, int]] = {}
    bands_seen = set()

    for fc in final_clusters:
        if fc.true_call_final is None:
            continue

        # Normalize callsign if alphabet provided
        if alphabet:
            call = normalize_call(fc.true_call_final, alphabet)
        else:
            call = fc.true_call_final.upper()

        # Get band from cluster key
        band = fc.cluster.key.band
        bands_seen.add(band)

        # Update global counts
        global_counts[call] = global_counts.get(call, 0) + 1

        # Update band-specific counts
        if band not in by_band:
            by_band[band] = {}
        by_band[band][call] = by_band[band].get(call, 0) + 1

    logger.info(f"Computed band priors for {len(global_counts)} callsigns across {len(bands_seen)} bands")

    return BandPriors(
        global_counts=global_counts,
        by_band=by_band,
        bands_seen=sorted(bands_seen)
    )


def freq_hz_to_khz(freq_hz: float) -> float:
    """Convert frequency from Hz to kHz."""
    return freq_hz / 1000.0


def get_segment_for_freq(
    freq_khz: float,
    band: str,
    segment_definitions: Dict[str, List[Tuple[float, float, str]]]
) -> Optional[str]:
    """Determine which segment a frequency belongs to.

    Args:
        freq_khz: Frequency in kHz
        band: Amateur band (e.g., "20m")
        segment_definitions: Segment definitions by band

    Returns:
        Segment name if found, None otherwise
    """
    if band not in segment_definitions:
        return None

    for start_khz, end_khz, segment_name in segment_definitions[band]:
        if start_khz <= freq_khz < end_khz:
            return segment_name

    return None


def compute_segment_priors(
    final_clusters: List[FinalCluster],
    segment_definitions: Optional[Dict[str, List[Tuple[float, float, str]]]] = None,
    alphabet: Optional[Alphabet] = None
) -> SegmentPriors:
    """Compute frequency segment callsign priors from final clusters.

    Args:
        final_clusters: List of FinalCluster objects with true_call_final
        segment_definitions: Optional custom segment definitions (uses defaults if None)
        alphabet: Optional alphabet for callsign normalization

    Returns:
        SegmentPriors object with per-segment counts
    """
    if segment_definitions is None:
        segment_definitions = DEFAULT_SEGMENTS

    by_segment: Dict[str, Dict[str, int]] = {}
    segment_info: Dict[str, Dict] = {}

    # Initialize segment info
    for band, segments in segment_definitions.items():
        for start_khz, end_khz, segment_name in segments:
            segment_info[segment_name] = {
                "band": band,
                "start_khz": start_khz,
                "end_khz": end_khz
            }
            by_segment[segment_name] = {}

    segments_used = set()

    for fc in final_clusters:
        if fc.true_call_final is None:
            continue

        # Normalize callsign if alphabet provided
        if alphabet:
            call = normalize_call(fc.true_call_final, alphabet)
        else:
            call = fc.true_call_final.upper()

        # Get band and average frequency
        band = fc.cluster.key.band
        spots = fc.cluster.spots
        if not spots:
            continue

        avg_freq_hz = sum(s.freq_hz for s in spots) / len(spots)
        freq_khz = freq_hz_to_khz(avg_freq_hz)

        # Find segment
        segment = get_segment_for_freq(freq_khz, band, segment_definitions)
        if segment is None:
            continue

        segments_used.add(segment)

        # Update segment counts
        by_segment[segment][call] = by_segment[segment].get(call, 0) + 1

    # Remove empty segments
    by_segment = {k: v for k, v in by_segment.items() if v}
    segment_info = {k: v for k, v in segment_info.items() if k in by_segment}

    logger.info(f"Computed segment priors for {len(segments_used)} segments")

    return SegmentPriors(
        by_segment=by_segment,
        segment_info=segment_info
    )


def export_band_priors(
    priors: BandPriors,
    output_path: Path
) -> None:
    """Export band priors to JSON file.

    Args:
        priors: BandPriors object
        output_path: Path to write JSON file
    """
    data = {
        "bands": priors.bands_seen,
        "global": priors.global_counts,
        "by_band": priors.by_band
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported band priors to {output_path}")
    logger.info(f"  Global: {len(priors.global_counts)} callsigns")
    for band in priors.bands_seen:
        if band in priors.by_band:
            logger.info(f"  {band}: {len(priors.by_band[band])} callsigns")


def export_segment_priors(
    priors: SegmentPriors,
    output_path: Path
) -> None:
    """Export segment priors to JSON file.

    Args:
        priors: SegmentPriors object
        output_path: Path to write JSON file
    """
    data = {
        "segments": priors.segment_info,
        "by_segment": priors.by_segment
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported segment priors to {output_path}")
    for segment_name, counts in priors.by_segment.items():
        logger.info(f"  {segment_name}: {len(counts)} callsigns")


def export_band_priors_simple(
    priors: BandPriors,
    output_path: Path,
    score_scale: float = 1.0,
    min_count: int = 1
) -> int:
    """Export band priors in simple text format for Go integration.

    Outputs a text file in the format expected by Go's LoadCallQualityPriors:
    CALL SCORE [FREQ_KHZ]

    For band priors, we output each call with a score derived from its count,
    optionally associated with a representative frequency for that band.

    Args:
        priors: BandPriors object
        output_path: Path to write text file
        score_scale: Scale factor for converting counts to scores
        min_count: Minimum count to include a callsign

    Returns:
        Number of entries exported
    """
    import math

    # Band center frequencies (approximate, for reference)
    band_center_khz = {
        "160m": 1850,
        "80m": 3550,
        "40m": 7025,
        "30m": 10125,
        "20m": 14050,
        "17m": 18100,
        "15m": 21050,
        "12m": 24920,
        "10m": 28050,
        "6m": 50100,
    }

    lines = []
    lines.append("# Band-specific call quality priors")
    lines.append("# Generated from RBN analysis")
    lines.append("# Format: CALL SCORE [FREQ_KHZ]")
    lines.append("")

    # First, output global priors (no frequency)
    lines.append("# Global priors (all bands)")
    for call, count in sorted(priors.global_counts.items(), key=lambda x: -x[1]):
        if count < min_count:
            continue
        # Log scale for score
        score = int(math.log(count + 1) * score_scale) + 1
        lines.append(f"{call} {score}")

    # Then output per-band priors with representative frequency
    for band in priors.bands_seen:
        if band not in priors.by_band:
            continue

        center_khz = band_center_khz.get(band)
        if center_khz is None:
            continue

        lines.append("")
        lines.append(f"# {band} priors")

        for call, count in sorted(priors.by_band[band].items(), key=lambda x: -x[1]):
            if count < min_count:
                continue
            score = int(math.log(count + 1) * score_scale) + 1
            lines.append(f"{call} {score} {center_khz}")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
        f.write("\n")

    entry_count = sum(1 for line in lines if line and not line.startswith("#"))
    logger.info(f"Exported {entry_count} entries to {output_path} (simple format)")

    return entry_count
