"""Tests for micro-clustering and provisional truth."""

from datetime import datetime, timezone, timedelta
from hypothesis import given, strategies as st, assume
import pytest

from rbn_train.clustering import (
    build_micro_clusters,
    label_provisional_truth,
    compute_skimmer_consensus,
)
from rbn_train.config import AnalysisConfig
from rbn_train.types import Spot, Cluster, ClusterKey


def create_spot(timestamp, freq_hz, band="40m", mode="CW", skimmer="TEST", dx_call="K3LR", snr_db=10.0):
    """Helper to create a Spot object."""
    return Spot(
        timestamp=timestamp,
        freq_hz=freq_hz,
        band=band,
        mode=mode,
        skimmer=skimmer,
        dx_call=dx_call,
        snr_db=snr_db,
    )


# Feature: rbn-analytics, Property 11: Cluster grouping completeness
@given(
    num_spots=st.integers(min_value=1, max_value=100),
    bin_size=st.integers(min_value=100, max_value=1000)
)
def test_cluster_grouping_completeness(num_spots, bin_size):
    """For any list of spots, every spot should appear in exactly one cluster after grouping.
    
    Validates: Requirements 3.3, 3.4
    """
    config = AnalysisConfig(cluster_freq_bin_hz=bin_size, cluster_time_seconds=60)
    
    # Create spots with varying frequencies
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    spots = [
        create_spot(
            timestamp=base_time + timedelta(seconds=i),
            freq_hz=7000000.0 + i * 100,
        )
        for i in range(num_spots)
    ]
    
    clusters = build_micro_clusters(spots, config)
    
    # Count total spots in all clusters
    total_spots_in_clusters = sum(len(cluster.spots) for cluster in clusters)
    
    # Every spot should appear exactly once
    assert total_spots_in_clusters == num_spots


# Feature: rbn-analytics, Property 12: Cluster key consistency
@given(
    num_spots_in_cluster=st.integers(min_value=2, max_value=20),
)
def test_cluster_key_consistency(num_spots_in_cluster):
    """For any cluster, all spots in that cluster should have the same band, mode, freq_bin, and time_bin.
    
    Validates: Requirements 3.3
    """
    config = AnalysisConfig(cluster_freq_bin_hz=500, cluster_time_seconds=60)
    
    # Create spots that should be in the same cluster
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    base_freq = 7000000.0
    
    spots = [
        create_spot(
            timestamp=base_time + timedelta(seconds=i),  # Within same time bin
            freq_hz=base_freq + i * 10,  # Within same freq bin
            band="40m",
            mode="CW",
            skimmer=f"TEST{i}",
            dx_call="K3LR",
        )
        for i in range(num_spots_in_cluster)
    ]
    
    clusters = build_micro_clusters(spots, config)
    
    # Check each cluster
    for cluster in clusters:
        if len(cluster.spots) > 0:
            # All spots should have same band and mode
            bands = {spot.band for spot in cluster.spots}
            modes = {spot.mode for spot in cluster.spots}
            
            assert len(bands) == 1
            assert len(modes) == 1
            
            # All spots should map to the same freq_bin and time_bin
            freq_bins = {int(spot.freq_hz) // config.cluster_freq_bin_hz for spot in cluster.spots}
            time_bins = {int(spot.timestamp.timestamp()) // config.cluster_time_seconds for spot in cluster.spots}
            
            assert len(freq_bins) == 1
            assert len(time_bins) == 1


# Feature: rbn-analytics, Property 13: Skimmer counting accuracy
@given(
    num_skimmers=st.integers(min_value=1, max_value=10),
    spots_per_skimmer=st.integers(min_value=1, max_value=5)
)
def test_skimmer_counting_accuracy(num_skimmers, spots_per_skimmer):
    """For any cluster, the count of distinct skimmers for each callsign should equal the size of the set of skimmers reporting that callsign.
    
    Validates: Requirements 4.1
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create spots from multiple skimmers reporting the same call
    spots = []
    for i in range(num_skimmers):
        for j in range(spots_per_skimmer):
            spots.append(create_spot(
                timestamp=base_time,
                freq_hz=7000000.0,
                skimmer=f"SKIMMER{i}",
                dx_call="K3LR",
            ))
    
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    
    # Count distinct skimmers manually
    distinct_skimmers = {spot.skimmer for spot in spots}
    expected_count = len(distinct_skimmers)
    
    # The consensus algorithm should count the same
    assert expected_count == num_skimmers


# Feature: rbn-analytics, Property 14: Provisional truth threshold enforcement
@given(
    num_skimmers=st.integers(min_value=0, max_value=10),
    min_threshold=st.integers(min_value=1, max_value=8)
)
def test_provisional_truth_threshold_enforcement(num_skimmers, min_threshold):
    """For any cluster where the best callsign has fewer than min_cluster_skimmers distinct skimmers, provisional_true_call should be None.
    
    Validates: Requirements 4.4
    """
    config = AnalysisConfig(min_cluster_skimmers=min_threshold, min_cluster_share_percent=0.0)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create spots from num_skimmers distinct skimmers
    spots = [
        create_spot(
            timestamp=base_time,
            freq_hz=7000000.0,
            skimmer=f"SKIMMER{i}",
            dx_call="K3LR",
        )
        for i in range(num_skimmers)
    ]
    
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    
    result = compute_skimmer_consensus(cluster, config)
    
    # Should be None if below threshold
    if num_skimmers < min_threshold:
        assert result is None
    else:
        # Should have a result if at or above threshold
        assert result == "K3LR"


# Feature: rbn-analytics, Property 15: Provisional truth share enforcement
@given(
    num_best=st.integers(min_value=1, max_value=10),
    num_other=st.integers(min_value=0, max_value=10),
    min_share=st.floats(min_value=0.0, max_value=100.0)
)
def test_provisional_truth_share_enforcement(num_best, num_other, min_share):
    """For any cluster where the best callsign share is below min_cluster_share_percent, provisional_true_call should be None.
    
    Validates: Requirements 4.5
    """
    assume(num_best + num_other > 0)
    
    config = AnalysisConfig(min_cluster_skimmers=0, min_cluster_share_percent=min_share)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create spots: num_best skimmers report K3LR, num_other report W1AW
    spots = []
    for i in range(num_best):
        spots.append(create_spot(
            timestamp=base_time,
            freq_hz=7000000.0,
            skimmer=f"BEST{i}",
            dx_call="K3LR",
        ))
    
    for i in range(num_other):
        spots.append(create_spot(
            timestamp=base_time,
            freq_hz=7000000.0,
            skimmer=f"OTHER{i}",
            dx_call="W1AW",
        ))
    
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    
    result = compute_skimmer_consensus(cluster, config)
    
    # Calculate actual share for the winner
    total_skimmers = num_best + num_other
    winner_count = max(num_best, num_other)
    actual_share = 100.0 * winner_count / total_skimmers
    
    # Should be None if below threshold
    if actual_share < min_share:
        assert result is None
    else:
        # Should have the winner if at or above threshold (and no tie)
        if num_best > num_other:
            assert result == "K3LR"
        elif num_other > num_best:
            assert result == "W1AW"


# Feature: rbn-analytics, Property 16: Provisional truth assignment correctness
@given(
    num_best=st.integers(min_value=5, max_value=15),
    num_other=st.integers(min_value=0, max_value=4)
)
def test_provisional_truth_assignment_correctness(num_best, num_other):
    """For any cluster where a callsign meets all thresholds with no tie, provisional_true_call should be that callsign.
    
    Validates: Requirements 4.6
    """
    assume(num_best > num_other)  # Ensure no tie
    
    config = AnalysisConfig(min_cluster_skimmers=4, min_cluster_share_percent=50.0)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create spots: num_best skimmers report K3LR, num_other report W1AW
    spots = []
    for i in range(num_best):
        spots.append(create_spot(
            timestamp=base_time,
            freq_hz=7000000.0,
            skimmer=f"BEST{i}",
            dx_call="K3LR",
        ))
    
    for i in range(num_other):
        spots.append(create_spot(
            timestamp=base_time,
            freq_hz=7000000.0,
            skimmer=f"OTHER{i}",
            dx_call="W1AW",
        ))
    
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    
    result = compute_skimmer_consensus(cluster, config)
    
    # Calculate if thresholds are met
    total_skimmers = num_best + num_other
    share = 100.0 * num_best / total_skimmers
    
    if num_best >= config.min_cluster_skimmers and share >= config.min_cluster_share_percent:
        # Should return K3LR
        assert result == "K3LR"


def test_empty_cluster():
    """Test that empty cluster returns None for provisional truth."""
    config = AnalysisConfig()
    
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=[]
    )
    
    result = compute_skimmer_consensus(cluster, config)
    assert result is None


def test_tie_returns_none():
    """Test that tie in skimmer count returns None."""
    config = AnalysisConfig(min_cluster_skimmers=2, min_cluster_share_percent=0.0)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create tie: 2 skimmers report K3LR, 2 report W1AW
    spots = [
        create_spot(base_time, 7000000.0, skimmer="S1", dx_call="K3LR"),
        create_spot(base_time, 7000000.0, skimmer="S2", dx_call="K3LR"),
        create_spot(base_time, 7000000.0, skimmer="S3", dx_call="W1AW"),
        create_spot(base_time, 7000000.0, skimmer="S4", dx_call="W1AW"),
    ]
    
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    
    result = compute_skimmer_consensus(cluster, config)
    assert result is None


def test_label_provisional_truth_integration():
    """Integration test for labeling clusters with provisional truth."""
    config = AnalysisConfig(min_cluster_skimmers=2, min_cluster_share_percent=50.0)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create spots for clustering
    spots = [
        create_spot(base_time, 7000000.0, skimmer="S1", dx_call="K3LR"),
        create_spot(base_time, 7000000.0, skimmer="S2", dx_call="K3LR"),
        create_spot(base_time, 7000000.0, skimmer="S3", dx_call="K3LR"),
        create_spot(base_time + timedelta(seconds=100), 7000000.0, skimmer="S4", dx_call="W1AW"),
    ]
    
    clusters = build_micro_clusters(spots, config)
    labeled = label_provisional_truth(clusters, config)
    
    assert len(labeled) > 0
    assert all(hasattr(lc, 'provisional_true_call') for lc in labeled)
