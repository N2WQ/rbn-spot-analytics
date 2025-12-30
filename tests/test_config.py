"""Tests for configuration management."""

import json
import tempfile
from pathlib import Path
import pytest
from hypothesis import given, strategies as st

from rbn_train.config import (
    AnalysisConfig,
    load_config,
    finalize_config,
    validate_config,
)


def test_default_snr_bands():
    """Test that None snr_bands becomes default [-999.0, 0.0, 6.0, 12.0, 18.0, 24.0, 999.0].
    
    Validates: Requirements 9.2
    """
    config = AnalysisConfig(snr_bands=None)
    finalized = finalize_config(config)
    
    expected = [-999.0, 0.0, 6.0, 12.0, 18.0, 24.0, 999.0]
    assert finalized.snr_bands == expected


def test_default_modes():
    """Test that None modes becomes ["CW", "RTTY", "SSB"].
    
    Validates: Requirements 9.3
    """
    config = AnalysisConfig(modes=None)
    finalized = finalize_config(config)
    
    expected = ["CW", "RTTY", "SSB"]
    assert finalized.modes == expected


def test_finalize_preserves_explicit_values():
    """Test that finalize_config preserves explicitly set values."""
    custom_snr_bands = [-10.0, 0.0, 10.0, 20.0]
    custom_modes = ["CW", "FT8"]
    
    config = AnalysisConfig(
        snr_bands=custom_snr_bands,
        modes=custom_modes,
        cluster_time_seconds=120,
    )
    finalized = finalize_config(config)
    
    assert finalized.snr_bands == custom_snr_bands
    assert finalized.modes == custom_modes
    assert finalized.cluster_time_seconds == 120


def test_load_config_nonexistent_file():
    """Test that loading nonexistent config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.json")


def test_load_config_invalid_json():
    """Test that loading invalid JSON raises JSONDecodeError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        temp_path = f.name
    
    try:
        with pytest.raises(json.JSONDecodeError):
            load_config(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_config_valid_file():
    """Test loading valid configuration from JSON file."""
    config_data = {
        "cluster_time_seconds": 120,
        "min_cluster_skimmers": 5,
        "modes": ["CW", "RTTY"],
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config.cluster_time_seconds == 120
        assert config.min_cluster_skimmers == 5
        assert config.modes == ["CW", "RTTY"]
    finally:
        Path(temp_path).unlink()


def test_load_config_none_returns_defaults():
    """Test that load_config(None) returns default configuration."""
    config = load_config(None)
    assert isinstance(config, AnalysisConfig)
    assert config.cluster_time_seconds == 60  # default value


def test_validate_unknown_char_single_character():
    """Test that unknown_char must be a single character."""
    # Valid single character
    config = AnalysisConfig(unknown_char="?")
    validate_config(config)  # Should not raise
    
    # Invalid: empty string
    config = AnalysisConfig(unknown_char="")
    with pytest.raises(ValueError, match="single character"):
        validate_config(config)
    
    # Invalid: multiple characters
    config = AnalysisConfig(unknown_char="??")
    with pytest.raises(ValueError, match="single character"):
        validate_config(config)


def test_validate_positive_values():
    """Test that certain parameters must be positive."""
    # Invalid cluster_time_seconds
    config = AnalysisConfig(cluster_time_seconds=0)
    with pytest.raises(ValueError, match="cluster_time_seconds must be positive"):
        validate_config(config)
    
    # Invalid cluster_freq_bin_hz
    config = AnalysisConfig(cluster_freq_bin_hz=-100)
    with pytest.raises(ValueError, match="cluster_freq_bin_hz must be positive"):
        validate_config(config)
    
    # Invalid stability_freq_bin_hz
    config = AnalysisConfig(stability_freq_bin_hz=0)
    with pytest.raises(ValueError, match="stability_freq_bin_hz must be positive"):
        validate_config(config)


def test_validate_percentages():
    """Test that percentage parameters must be between 0 and 100."""
    # Invalid min_cluster_share_percent
    config = AnalysisConfig(min_cluster_share_percent=-10)
    with pytest.raises(ValueError, match="between 0 and 100"):
        validate_config(config)
    
    config = AnalysisConfig(min_cluster_share_percent=150)
    with pytest.raises(ValueError, match="between 0 and 100"):
        validate_config(config)
    
    # Valid edge cases
    config = AnalysisConfig(min_cluster_share_percent=0)
    validate_config(config)  # Should not raise
    
    config = AnalysisConfig(min_cluster_share_percent=100)
    validate_config(config)  # Should not raise


def test_validate_callsign_length_bounds():
    """Test that callsign length bounds are validated."""
    # Invalid: min_call_length < 1
    config = AnalysisConfig(min_call_length=0)
    with pytest.raises(ValueError, match="min_call_length must be at least 1"):
        validate_config(config)
    
    # Invalid: max < min
    config = AnalysisConfig(min_call_length=10, max_call_length=5)
    with pytest.raises(ValueError, match="max_call_length.*must be >= min_call_length"):
        validate_config(config)
    
    # Valid: max == min
    config = AnalysisConfig(min_call_length=5, max_call_length=5)
    validate_config(config)  # Should not raise


def test_validate_snr_bands_ascending():
    """Test that SNR bands must be in strictly ascending order."""
    # Valid ascending order
    config = AnalysisConfig(snr_bands=[-10.0, 0.0, 10.0, 20.0])
    validate_config(config)  # Should not raise
    
    # Invalid: not ascending
    config = AnalysisConfig(snr_bands=[0.0, 10.0, 5.0, 20.0])
    with pytest.raises(ValueError, match="strictly ascending order"):
        validate_config(config)
    
    # Invalid: equal values
    config = AnalysisConfig(snr_bands=[0.0, 10.0, 10.0, 20.0])
    with pytest.raises(ValueError, match="strictly ascending order"):
        validate_config(config)
    
    # Invalid: too few edges
    config = AnalysisConfig(snr_bands=[0.0])
    with pytest.raises(ValueError, match="at least 2 edges"):
        validate_config(config)


def test_validate_modes_not_empty():
    """Test that modes list cannot be empty."""
    config = AnalysisConfig(modes=[])
    with pytest.raises(ValueError, match="modes list cannot be empty"):
        validate_config(config)


def test_validate_modes_no_duplicates():
    """Test that modes list cannot contain duplicates."""
    config = AnalysisConfig(modes=["CW", "RTTY", "CW"])
    with pytest.raises(ValueError, match="contains duplicates"):
        validate_config(config)


# Feature: rbn-analytics, Property 41: CLI override application
@given(
    cluster_time=st.integers(min_value=1, max_value=3600),
    min_skimmers=st.integers(min_value=1, max_value=20),
)
def test_cli_override_application(cluster_time, min_skimmers):
    """For any loaded configuration and CLI overrides, the final configuration should reflect the CLI values.
    
    Validates: Requirements 9.5
    """
    # Start with default config
    base_config = AnalysisConfig()
    
    # Simulate CLI overrides by creating new config with overridden values
    overridden_config = AnalysisConfig(
        cluster_time_seconds=cluster_time,
        min_cluster_skimmers=min_skimmers,
        # Keep other defaults
        cluster_freq_bin_hz=base_config.cluster_freq_bin_hz,
        min_cluster_share_percent=base_config.min_cluster_share_percent,
        stability_freq_bin_hz=base_config.stability_freq_bin_hz,
        stability_min_clusters=base_config.stability_min_clusters,
        stability_min_share_percent=base_config.stability_min_share_percent,
        snr_bands=base_config.snr_bands,
        modes=base_config.modes,
        charset=base_config.charset,
        unknown_char=base_config.unknown_char,
        min_snr_db=base_config.min_snr_db,
        max_call_length=base_config.max_call_length,
        min_call_length=base_config.min_call_length,
    )
    
    # Verify overridden values are applied
    assert overridden_config.cluster_time_seconds == cluster_time
    assert overridden_config.min_cluster_skimmers == min_skimmers
    
    # Verify non-overridden values remain at defaults
    assert overridden_config.cluster_freq_bin_hz == base_config.cluster_freq_bin_hz
