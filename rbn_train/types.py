"""Core data types for RBN analysis pipeline."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Spot:
    """A single RBN observation record.
    
    Represents one decoded signal observation from an RBN skimmer station,
    including timestamp, frequency, band, mode, skimmer ID, decoded callsign, and SNR.
    """
    timestamp: datetime
    freq_hz: float
    band: str
    mode: str
    skimmer: str
    dx_call: str
    snr_db: float


@dataclass(frozen=True)
class ClusterKey:
    """Immutable key for grouping spots into micro-clusters.
    
    Combines band, mode, frequency bin, and time bin to identify
    observations of the same signal within a small time and frequency window.
    """
    band: str
    mode: str
    freq_bin: int
    time_bin: int


@dataclass
class Cluster:
    """A collection of spots grouped by ClusterKey.
    
    Represents multiple observations of the same signal from different
    skimmers within a micro-cluster window.
    """
    key: ClusterKey
    spots: List[Spot]


@dataclass
class LabeledCluster:
    """A cluster with an assigned provisional true callsign.
    
    The provisional_true_call is determined by skimmer consensus
    within the micro-cluster.
    """
    cluster: Cluster
    provisional_true_call: Optional[str]


@dataclass(frozen=True)
class StabilityKey:
    """Immutable key for grouping clusters into stability bins.
    
    Combines band, mode, and a coarser frequency bin to validate
    callsigns across multiple micro-clusters.
    """
    band: str
    mode: str
    freq_bin: int


@dataclass
class StableBin:
    """A collection of clusters in a stability bin with validated callsign.
    
    The stable_call is determined by consistency across multiple
    micro-clusters in the larger frequency bin.
    """
    key: StabilityKey
    stable_call: Optional[str]


@dataclass
class FinalCluster:
    """A cluster with refined true callsign after stability analysis.
    
    The true_call_final is the result of combining provisional truth
    with stability validation and Levenshtein distance matching.
    """
    cluster: Cluster
    true_call_final: Optional[str]
