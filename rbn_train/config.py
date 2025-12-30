"""Configuration management for RBN analysis."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AnalysisConfig:
    """Configuration parameters for RBN analysis pipeline.

    Attributes:
        cluster_time_seconds: Time window in seconds for micro-clustering
        cluster_freq_bin_hz: Frequency bin size in Hz for micro-clustering
        min_cluster_skimmers: Minimum distinct skimmers required for provisional truth
        min_cluster_share_percent: Minimum percentage share for provisional truth
        stability_freq_bin_hz: Frequency bin size in Hz for stability analysis
        stability_min_clusters: Minimum clusters required for stable call
        stability_min_share_percent: Minimum percentage share for stable call
        snr_bands: SNR band edges for confusion model (None = use defaults)
        modes: List of modes to process (None = use defaults)
        charset: Character set for callsign alphabet
        unknown_char: Character to use for unknown/invalid characters
        min_snr_db: Minimum SNR threshold for filtering spots
        max_call_length: Maximum callsign length
        min_call_length: Minimum callsign length
        output_spotter_reliability: Whether to output spotter reliability analysis
        output_band_priors: Whether to output band-specific priors
        output_segment_priors: Whether to output frequency segment priors
        min_spotter_spots: Minimum spots for spotter reliability inclusion
        output_simple_formats: Whether to output Go-compatible simple text formats
    """
    # Micro-cluster parameters
    cluster_time_seconds: int = 60
    cluster_freq_bin_hz: int = 500
    min_cluster_skimmers: int = 4
    min_cluster_share_percent: float = 70.0

    # Stability parameters
    stability_freq_bin_hz: int = 1000
    stability_min_clusters: int = 5
    stability_min_share_percent: float = 80.0

    # Confusion model parameters
    snr_bands: Optional[List[float]] = None
    modes: Optional[List[str]] = None

    # Character set
    charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/"
    unknown_char: str = "?"

    # Data filtering
    min_snr_db: float = -999.0
    max_call_length: int = 16
    min_call_length: int = 2

    # Extended analysis outputs
    output_spotter_reliability: bool = True
    output_band_priors: bool = True
    output_segment_priors: bool = True
    min_spotter_spots: int = 1000
    output_simple_formats: bool = True

    # Parallelization
    workers: int = 1  # Number of worker processes (1 = sequential, 0 = auto-detect)
    parallel_chunk_size: int = 1000  # Items per chunk for parallel processing


def load_config(path: Optional[str]) -> AnalysisConfig:
    """Load configuration from JSON file.
    
    Args:
        path: Path to JSON configuration file, or None for defaults
        
    Returns:
        AnalysisConfig object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If config contains invalid parameter types
    """
    if path is None:
        return AnalysisConfig()
    
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in configuration file: {path}",
            e.doc,
            e.pos
        )
    
    # Create config with loaded values
    try:
        return AnalysisConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration parameters: {e}")


def finalize_config(config: AnalysisConfig) -> AnalysisConfig:
    """Apply default values for None fields.

    Args:
        config: Configuration object to finalize

    Returns:
        New AnalysisConfig with defaults applied
    """
    # Create a copy with defaults applied
    snr_bands = config.snr_bands
    if snr_bands is None:
        snr_bands = [-999.0, 0.0, 6.0, 12.0, 18.0, 24.0, 999.0]

    modes = config.modes
    if modes is None:
        modes = ["CW", "RTTY"]  # RBN only supports CW and RTTY

    # Resolve worker count (0 = auto-detect)
    from rbn_train.parallel import get_worker_count
    workers = get_worker_count(config.workers) if config.workers == 0 else config.workers

    return AnalysisConfig(
        cluster_time_seconds=config.cluster_time_seconds,
        cluster_freq_bin_hz=config.cluster_freq_bin_hz,
        min_cluster_skimmers=config.min_cluster_skimmers,
        min_cluster_share_percent=config.min_cluster_share_percent,
        stability_freq_bin_hz=config.stability_freq_bin_hz,
        stability_min_clusters=config.stability_min_clusters,
        stability_min_share_percent=config.stability_min_share_percent,
        snr_bands=snr_bands,
        modes=modes,
        charset=config.charset,
        unknown_char=config.unknown_char,
        min_snr_db=config.min_snr_db,
        max_call_length=config.max_call_length,
        min_call_length=config.min_call_length,
        output_spotter_reliability=config.output_spotter_reliability,
        output_band_priors=config.output_band_priors,
        output_segment_priors=config.output_segment_priors,
        min_spotter_spots=config.min_spotter_spots,
        output_simple_formats=config.output_simple_formats,
        workers=workers,
        parallel_chunk_size=config.parallel_chunk_size,
    )


def validate_config(config: AnalysisConfig) -> None:
    """Validate configuration constraints.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration violates constraints
    """
    # Validate unknown_char is single character
    if len(config.unknown_char) != 1:
        raise ValueError(
            f"unknown_char must be a single character, got: '{config.unknown_char}'"
        )
    
    # Validate positive values
    if config.cluster_time_seconds <= 0:
        raise ValueError(f"cluster_time_seconds must be positive, got: {config.cluster_time_seconds}")
    
    if config.cluster_freq_bin_hz <= 0:
        raise ValueError(f"cluster_freq_bin_hz must be positive, got: {config.cluster_freq_bin_hz}")
    
    if config.stability_freq_bin_hz <= 0:
        raise ValueError(f"stability_freq_bin_hz must be positive, got: {config.stability_freq_bin_hz}")
    
    if config.min_cluster_skimmers < 0:
        raise ValueError(f"min_cluster_skimmers must be non-negative, got: {config.min_cluster_skimmers}")
    
    if config.stability_min_clusters < 0:
        raise ValueError(f"stability_min_clusters must be non-negative, got: {config.stability_min_clusters}")
    
    # Validate percentages
    if not (0 <= config.min_cluster_share_percent <= 100):
        raise ValueError(f"min_cluster_share_percent must be between 0 and 100, got: {config.min_cluster_share_percent}")
    
    if not (0 <= config.stability_min_share_percent <= 100):
        raise ValueError(f"stability_min_share_percent must be between 0 and 100, got: {config.stability_min_share_percent}")
    
    # Validate callsign length bounds
    if config.min_call_length < 1:
        raise ValueError(f"min_call_length must be at least 1, got: {config.min_call_length}")
    
    if config.max_call_length < config.min_call_length:
        raise ValueError(
            f"max_call_length ({config.max_call_length}) must be >= min_call_length ({config.min_call_length})"
        )
    
    # Validate SNR bands if provided
    if config.snr_bands is not None:
        if len(config.snr_bands) < 2:
            raise ValueError(f"snr_bands must have at least 2 edges, got: {len(config.snr_bands)}")
        
        # Check that bands are in ascending order
        for i in range(len(config.snr_bands) - 1):
            if config.snr_bands[i] >= config.snr_bands[i + 1]:
                raise ValueError(
                    f"snr_bands must be in strictly ascending order, "
                    f"but {config.snr_bands[i]} >= {config.snr_bands[i + 1]} at index {i}"
                )
    
    # Validate modes if provided
    if config.modes is not None:
        if len(config.modes) == 0:
            raise ValueError("modes list cannot be empty")

        # Check for duplicates
        if len(config.modes) != len(set(config.modes)):
            raise ValueError(f"modes list contains duplicates: {config.modes}")

    # Validate extended analysis parameters
    if config.min_spotter_spots < 1:
        raise ValueError(f"min_spotter_spots must be at least 1, got: {config.min_spotter_spots}")

    # Validate parallelization parameters
    if config.workers < 0:
        raise ValueError(f"workers must be non-negative, got: {config.workers}")

    if config.parallel_chunk_size < 1:
        raise ValueError(f"parallel_chunk_size must be at least 1, got: {config.parallel_chunk_size}")
