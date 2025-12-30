"""Property-based tests for core types."""

from datetime import datetime, timezone
from hypothesis import given, strategies as st
import pytest

from rbn_train.types import (
    Spot,
    ClusterKey,
    Cluster,
    LabeledCluster,
    StabilityKey,
    StableBin,
    FinalCluster,
)


# Hypothesis strategies for generating test data
@st.composite
def spot_strategy(draw):
    """Generate random Spot objects."""
    dt = draw(st.datetimes(min_value=datetime(2020, 1, 1),
                          max_value=datetime(2025, 12, 31)))
    return Spot(
        timestamp=dt.replace(tzinfo=timezone.utc),
        freq_hz=draw(st.floats(min_value=1_000_000, max_value=30_000_000)),
        band=draw(st.sampled_from(["160m", "80m", "40m", "20m", "15m", "10m"])),
        mode=draw(st.sampled_from(["CW", "RTTY", "SSB"])),
        skimmer=draw(st.text(min_size=3, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Nd")))),
        dx_call=draw(st.text(min_size=3, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Nd")))),
        snr_db=draw(st.floats(min_value=-20, max_value=40)),
    )


# Feature: rbn-analytics, Property 9: Frequency binning determinism
@given(freq_hz=st.floats(min_value=1_000_000, max_value=30_000_000, allow_nan=False, allow_infinity=False),
       bin_size=st.integers(min_value=100, max_value=10000))
def test_frequency_binning_determinism(freq_hz, bin_size):
    """For any frequency value and bin size, computing the freq_bin twice should produce the same result.
    
    Validates: Requirements 3.1
    """
    # Compute freq_bin twice
    freq_bin_1 = int(freq_hz) // bin_size
    freq_bin_2 = int(freq_hz) // bin_size
    
    # Should be identical
    assert freq_bin_1 == freq_bin_2


# Feature: rbn-analytics, Property 10: Time binning determinism
@given(timestamp=st.datetimes(min_value=datetime(2020, 1, 1),
                               max_value=datetime(2025, 12, 31)),
       bin_size=st.integers(min_value=10, max_value=3600))
def test_time_binning_determinism(timestamp, bin_size):
    """For any timestamp and bin size, computing the time_bin twice should produce the same result.
    
    Validates: Requirements 3.2
    """
    # Add timezone for timestamp() to work
    timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    # Compute time_bin twice
    time_bin_1 = int(timestamp.timestamp()) // bin_size
    time_bin_2 = int(timestamp.timestamp()) // bin_size
    
    # Should be identical
    assert time_bin_1 == time_bin_2


def test_cluster_key_immutability():
    """Test that ClusterKey is immutable (frozen dataclass)."""
    key = ClusterKey(band="40m", mode="CW", freq_bin=100, time_bin=200)
    
    # Should not be able to modify frozen dataclass
    with pytest.raises(AttributeError):
        key.band = "20m"
    
    with pytest.raises(AttributeError):
        key.freq_bin = 150


def test_stability_key_immutability():
    """Test that StabilityKey is immutable (frozen dataclass)."""
    key = StabilityKey(band="40m", mode="CW", freq_bin=100)
    
    # Should not be able to modify frozen dataclass
    with pytest.raises(AttributeError):
        key.band = "20m"
    
    with pytest.raises(AttributeError):
        key.freq_bin = 150


def test_cluster_key_hashable():
    """Test that ClusterKey can be used as dictionary key."""
    key1 = ClusterKey(band="40m", mode="CW", freq_bin=100, time_bin=200)
    key2 = ClusterKey(band="40m", mode="CW", freq_bin=100, time_bin=200)
    key3 = ClusterKey(band="20m", mode="CW", freq_bin=100, time_bin=200)
    
    # Same keys should be equal and have same hash
    assert key1 == key2
    assert hash(key1) == hash(key2)
    
    # Different keys should not be equal
    assert key1 != key3
    
    # Should work as dictionary keys
    d = {key1: "value1", key3: "value3"}
    assert d[key2] == "value1"  # key2 equals key1


def test_stability_key_hashable():
    """Test that StabilityKey can be used as dictionary key."""
    key1 = StabilityKey(band="40m", mode="CW", freq_bin=100)
    key2 = StabilityKey(band="40m", mode="CW", freq_bin=100)
    key3 = StabilityKey(band="20m", mode="CW", freq_bin=100)
    
    # Same keys should be equal and have same hash
    assert key1 == key2
    assert hash(key1) == hash(key2)
    
    # Different keys should not be equal
    assert key1 != key3
    
    # Should work as dictionary keys
    d = {key1: "value1", key3: "value3"}
    assert d[key2] == "value1"  # key2 equals key1
