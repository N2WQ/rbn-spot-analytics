"""RBN Callsign Correction Analytics Tool

An offline Python tool that analyzes large volumes of RBN (Reverse Beacon Network)
spot data from major CW contests to infer true callsigns and learn decoder error patterns.

Extended Outputs:
- confusion_model.json: Character-level error statistics by mode and SNR band
- priors.json: Global callsign frequency counts
- spotter_reliability.json: Per-skimmer accuracy metrics
- band_priors.json: Band-specific callsign frequency counts
- segment_priors.json: Frequency segment callsign counts
- spotter_reliability.txt: Simple format for Go integration
- call_quality_priors.txt: Simple format for Go integration
"""

__version__ = "0.2.0"

from rbn_train.types import (
    Spot,
    ClusterKey,
    Cluster,
    LabeledCluster,
    StabilityKey,
    StableBin,
    FinalCluster,
)

from rbn_train.reliability import (
    SpotterStats,
    SpotterReliability,
    compute_spotter_reliability,
    export_spotter_reliability,
    export_spotter_reliability_simple,
)

from rbn_train.band_priors import (
    BandPriors,
    SegmentPriors,
    compute_band_priors,
    compute_segment_priors,
    export_band_priors,
    export_segment_priors,
    export_band_priors_simple,
)

__all__ = [
    # Core types
    "Spot",
    "ClusterKey",
    "Cluster",
    "LabeledCluster",
    "StabilityKey",
    "StableBin",
    "FinalCluster",
    # Reliability types and functions
    "SpotterStats",
    "SpotterReliability",
    "compute_spotter_reliability",
    "export_spotter_reliability",
    "export_spotter_reliability_simple",
    # Band/segment priors types and functions
    "BandPriors",
    "SegmentPriors",
    "compute_band_priors",
    "compute_segment_priors",
    "export_band_priors",
    "export_segment_priors",
    "export_band_priors_simple",
]
