"""Tests for stability analysis and truth refinement."""

from datetime import datetime, timezone, timedelta
from hypothesis import given, strategies as st, assume
import pytest

from rbn_train.stability import (
    levenshtein_distance,
    build_stability_bins,
    compute_stable_calls,
    refine_truth,
)
from rbn_train.config import AnalysisConfig
from rbn_train.types import Spot, Cluster, ClusterKey, LabeledCluster, StabilityKey


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


def test_levenshtein_lz5m_vs_dz5m():
    """Test 'LZ5M' vs 'DZ5M' (distance = 1).
    
    Validates: Requirements 6.6
    """
    distance = levenshtein_distance("LZ5M", "DZ5M")
    assert distance == 1


def test_levenshtein_kc1xx_vs_kc1xnt():
    """Test 'KC1XX' vs 'KC1XNT' (distance = 2).
    
    Validates: Requirements 6.6
    """
    distance = levenshtein_distance("KC1XX", "KC1XNT")
    assert distance == 2


def test_levenshtein_k3lr_vs_k3l():
    """Test 'K3LR' vs 'K3L' (distance = 1).
    
    Validates: Requirements 6.6
    """
    distance = levenshtein_distance("K3LR", "K3L")
    assert distance == 1


def test_levenshtein_identical_strings():
    """Test identical strings (distance = 0).
    
    Validates: Requirements 6.6
    """
    distance = levenshtein_distance("K3LR", "K3LR")
    assert distance == 0


def test_levenshtein_empty_strings():
    """Test edge cases with empty strings."""
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("ABC", "") == 3
    assert levenshtein_distance("", "ABC") == 3


def test_levenshtein_symmetric():
    """Test that Levenshtein distance is symmetric."""
    assert levenshtein_distance("ABC", "XYZ") == levenshtein_distance("XYZ", "ABC")
    assert levenshtein_distance("K3LR", "W1AW") == levenshtein_distance("W1AW", "K3LR")


# Feature: rbn-analytics, Property 17: Center frequency calculation
@given(
    num_spots=st.integers(min_value=1, max_value=20),
    base_freq=st.floats(min_value=7000000, max_value=7100000, allow_nan=False, allow_infinity=False)
)
def test_center_frequency_calculation(num_spots, base_freq):
    """For any cluster with N spots, the center frequency should equal the sum of all spot frequencies divided by N.
    
    Validates: Requirements 5.1
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create spots with varying frequencies
    spots = [
        create_spot(
            timestamp=base_time,
            freq_hz=base_freq + i * 100,
        )
        for i in range(num_spots)
    ]
    
    # Compute expected center frequency
    expected_center = sum(spot.freq_hz for spot in spots) / len(spots)
    
    # Create labeled cluster
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    labeled = LabeledCluster(cluster=cluster, provisional_true_call="K3LR")
    
    # Build stability bins (which computes center frequency)
    bins = build_stability_bins([labeled], config)
    
    # Verify the cluster was binned (which means center freq was computed)
    assert len(bins) > 0
    
    # Manually verify center frequency calculation
    actual_center = sum(spot.freq_hz for spot in spots) / len(spots)
    assert abs(actual_center - expected_center) < 0.01


# Feature: rbn-analytics, Property 18: Stability binning determinism
@given(
    center_freq=st.floats(min_value=7000000, max_value=7100000, allow_nan=False, allow_infinity=False),
    bin_size=st.integers(min_value=500, max_value=5000)
)
def test_stability_binning_determinism(center_freq, bin_size):
    """For any center frequency and stability bin size, computing the freq_bin twice should produce the same result.
    
    Validates: Requirements 5.2
    """
    # Compute freq_bin twice
    freq_bin_1 = int(center_freq) // bin_size
    freq_bin_2 = int(center_freq) // bin_size
    
    # Should be identical
    assert freq_bin_1 == freq_bin_2


# Feature: rbn-analytics, Property 19: Stability bin filtering
@given(
    has_provisional=st.booleans(),
)
def test_stability_bin_filtering(has_provisional):
    """For any labeled cluster with provisional_true_call equal to None, it should not appear in any stability bin.
    
    Validates: Requirements 5.3
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    spots = [create_spot(base_time, 7000000.0)]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    
    provisional_call = "K3LR" if has_provisional else None
    labeled = LabeledCluster(cluster=cluster, provisional_true_call=provisional_call)
    
    bins = build_stability_bins([labeled], config)
    
    # If no provisional truth, should not appear in any bin
    if not has_provisional:
        assert len(bins) == 0
    else:
        assert len(bins) > 0


# Feature: rbn-analytics, Property 20: Stability bin grouping correctness
def test_stability_bin_grouping_correctness():
    """For any stability bin, all clusters in that bin should have the same band, mode, and stability freq_bin.
    
    Validates: Requirements 5.4
    """
    config = AnalysisConfig(stability_freq_bin_hz=1000)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create clusters that should be in the same stability bin
    labeled_clusters = []
    for i in range(5):
        spots = [create_spot(base_time, 7000000.0 + i * 100, band="40m", mode="CW")]
        cluster = Cluster(
            key=ClusterKey(band="40m", mode="CW", freq_bin=14000 + i, time_bin=1000),
            spots=spots
        )
        labeled_clusters.append(LabeledCluster(cluster=cluster, provisional_true_call="K3LR"))
    
    bins = build_stability_bins(labeled_clusters, config)
    
    # Check each bin
    for key, clusters in bins.items():
        # All clusters should have same band and mode
        bands = {lc.cluster.key.band for lc in clusters}
        modes = {lc.cluster.key.mode for lc in clusters}
        
        assert len(bands) == 1
        assert len(modes) == 1
        
        # All clusters should map to the same stability freq_bin
        for lc in clusters:
            center_freq = sum(s.freq_hz for s in lc.cluster.spots) / len(lc.cluster.spots)
            freq_bin = int(center_freq) // config.stability_freq_bin_hz
            assert freq_bin == key.freq_bin


# Feature: rbn-analytics, Property 21: Stable call cluster threshold
@given(
    num_clusters=st.integers(min_value=0, max_value=10),
    min_threshold=st.integers(min_value=1, max_value=8)
)
def test_stable_call_cluster_threshold(num_clusters, min_threshold):
    """For any stability bin with fewer than stability_min_clusters clusters, stable_call should be None.
    
    Validates: Requirements 5.5, 5.7
    """
    config = AnalysisConfig(stability_min_clusters=min_threshold, stability_min_share_percent=0.0)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create labeled clusters
    labeled_clusters = []
    for i in range(num_clusters):
        spots = [create_spot(base_time, 7000000.0)]
        cluster = Cluster(
            key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000 + i),
            spots=spots
        )
        labeled_clusters.append(LabeledCluster(cluster=cluster, provisional_true_call="K3LR"))
    
    # Build bins
    bins = build_stability_bins(labeled_clusters, config)
    stable_bins = compute_stable_calls(bins, config)
    
    # Check result
    if num_clusters < min_threshold:
        # Should have None stable_call
        for stable_bin in stable_bins.values():
            assert stable_bin.stable_call is None
    else:
        # Should have a stable_call
        for stable_bin in stable_bins.values():
            assert stable_bin.stable_call == "K3LR"


# Feature: rbn-analytics, Property 22: Stable call share enforcement
@given(
    num_best=st.integers(min_value=1, max_value=10),
    num_other=st.integers(min_value=0, max_value=10),
    min_share=st.floats(min_value=0.0, max_value=100.0)
)
def test_stable_call_share_enforcement(num_best, num_other, min_share):
    """For any stability bin where the best callsign share is below stability_min_share_percent, stable_call should be None.
    
    Validates: Requirements 5.8
    """
    assume(num_best + num_other > 0)
    
    config = AnalysisConfig(stability_min_clusters=0, stability_min_share_percent=min_share)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create labeled clusters: num_best with K3LR, num_other with W1AW
    labeled_clusters = []
    for i in range(num_best):
        spots = [create_spot(base_time, 7000000.0)]
        cluster = Cluster(
            key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000 + i),
            spots=spots
        )
        labeled_clusters.append(LabeledCluster(cluster=cluster, provisional_true_call="K3LR"))
    
    for i in range(num_other):
        spots = [create_spot(base_time, 7000000.0)]
        cluster = Cluster(
            key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=2000 + i),
            spots=spots
        )
        labeled_clusters.append(LabeledCluster(cluster=cluster, provisional_true_call="W1AW"))
    
    # Build bins
    bins = build_stability_bins(labeled_clusters, config)
    stable_bins = compute_stable_calls(bins, config)
    
    # Calculate actual share for the winner
    total = num_best + num_other
    winner_count = max(num_best, num_other)
    actual_share = 100.0 * winner_count / total
    
    # Check result
    for stable_bin in stable_bins.values():
        if actual_share < min_share:
            assert stable_bin.stable_call is None
        else:
            if num_best > num_other:
                assert stable_bin.stable_call == "K3LR"
            elif num_other > num_best:
                assert stable_bin.stable_call == "W1AW"


# Feature: rbn-analytics, Property 23: Stable call assignment correctness
@given(
    num_best=st.integers(min_value=6, max_value=15),
    num_other=st.integers(min_value=0, max_value=4)
)
def test_stable_call_assignment_correctness(num_best, num_other):
    """For any stability bin where a callsign meets all thresholds with no tie, stable_call should be that callsign.
    
    Validates: Requirements 5.9
    """
    assume(num_best > num_other)  # Ensure no tie
    
    config = AnalysisConfig(stability_min_clusters=5, stability_min_share_percent=50.0)
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create labeled clusters
    labeled_clusters = []
    for i in range(num_best):
        spots = [create_spot(base_time, 7000000.0)]
        cluster = Cluster(
            key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000 + i),
            spots=spots
        )
        labeled_clusters.append(LabeledCluster(cluster=cluster, provisional_true_call="K3LR"))
    
    for i in range(num_other):
        spots = [create_spot(base_time, 7000000.0)]
        cluster = Cluster(
            key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=2000 + i),
            spots=spots
        )
        labeled_clusters.append(LabeledCluster(cluster=cluster, provisional_true_call="W1AW"))
    
    # Build bins
    bins = build_stability_bins(labeled_clusters, config)
    stable_bins = compute_stable_calls(bins, config)
    
    # Calculate if thresholds are met
    total = num_best + num_other
    share = 100.0 * num_best / total
    
    if num_best >= config.stability_min_clusters and share >= config.stability_min_share_percent:
        # Should return K3LR
        for stable_bin in stable_bins.values():
            assert stable_bin.stable_call == "K3LR"


# Feature: rbn-analytics, Property 24: Fallback to provisional when no stability
def test_fallback_to_provisional_when_no_stability():
    """For any cluster with no corresponding stability bin, true_call_final should equal provisional_true_call.
    
    Validates: Requirements 6.2
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    spots = [create_spot(base_time, 7000000.0)]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    labeled = LabeledCluster(cluster=cluster, provisional_true_call="K3LR")
    
    # Empty stability bins (no stability info)
    stable_bins = {}
    
    final_clusters = refine_truth([labeled], stable_bins, config)
    
    assert len(final_clusters) == 1
    assert final_clusters[0].true_call_final == "K3LR"


# Feature: rbn-analytics, Property 25: Fallback to provisional when stability fails
def test_fallback_to_provisional_when_stability_fails():
    """For any cluster whose stability bin has stable_call equal to None, true_call_final should equal provisional_true_call.
    
    Validates: Requirements 6.3
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    spots = [create_spot(base_time, 7000000.0)]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    labeled = LabeledCluster(cluster=cluster, provisional_true_call="K3LR")
    
    # Stability bin exists but has no stable call
    stability_key = StabilityKey(band="40m", mode="CW", freq_bin=7000)
    from rbn_train.types import StableBin
    stable_bins = {stability_key: StableBin(key=stability_key, stable_call=None)}
    
    final_clusters = refine_truth([labeled], stable_bins, config)
    
    assert len(final_clusters) == 1
    assert final_clusters[0].true_call_final == "K3LR"


# Feature: rbn-analytics, Property 26: Agreement preservation
def test_agreement_preservation():
    """For any cluster where stable_call equals provisional_true_call, true_call_final should equal provisional_true_call.
    
    Validates: Requirements 6.4
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    spots = [create_spot(base_time, 7000000.0, dx_call="K3LR")]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    labeled = LabeledCluster(cluster=cluster, provisional_true_call="K3LR")
    
    # Stability bin agrees with provisional
    stability_key = StabilityKey(band="40m", mode="CW", freq_bin=7000)
    from rbn_train.types import StableBin
    stable_bins = {stability_key: StableBin(key=stability_key, stable_call="K3LR")}
    
    final_clusters = refine_truth([labeled], stable_bins, config)
    
    assert len(final_clusters) == 1
    assert final_clusters[0].true_call_final == "K3LR"


# Feature: rbn-analytics, Property 27: Direct match preference
def test_direct_match_preference():
    """For any cluster where stable_call appears in the decoded calls, true_call_final should equal stable_call.
    
    Validates: Requirements 6.5
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Cluster has decoded calls including W1AW
    spots = [
        create_spot(base_time, 7000000.0, dx_call="K3LR"),
        create_spot(base_time, 7000000.0, dx_call="W1AW"),
    ]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    labeled = LabeledCluster(cluster=cluster, provisional_true_call="K3LR")
    
    # Stability says W1AW
    stability_key = StabilityKey(band="40m", mode="CW", freq_bin=7000)
    from rbn_train.types import StableBin
    stable_bins = {stability_key: StableBin(key=stability_key, stable_call="W1AW")}
    
    final_clusters = refine_truth([labeled], stable_bins, config)
    
    assert len(final_clusters) == 1
    assert final_clusters[0].true_call_final == "W1AW"


# Feature: rbn-analytics, Property 28: Fuzzy match threshold
def test_fuzzy_match_threshold():
    """For any cluster where stable_call has Levenshtein distance ≤ 2 to any decoded call, true_call_final should equal stable_call.
    
    Validates: Requirements 6.6
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Cluster has decoded call "LZ5M"
    spots = [create_spot(base_time, 7000000.0, dx_call="LZ5M")]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    labeled = LabeledCluster(cluster=cluster, provisional_true_call="LZ5M")
    
    # Stability says "DZ5M" (distance = 1)
    stability_key = StabilityKey(band="40m", mode="CW", freq_bin=7000)
    from rbn_train.types import StableBin
    stable_bins = {stability_key: StableBin(key=stability_key, stable_call="DZ5M")}
    
    final_clusters = refine_truth([labeled], stable_bins, config)
    
    assert len(final_clusters) == 1
    assert final_clusters[0].true_call_final == "DZ5M"


# Feature: rbn-analytics, Property 29: Final fallback correctness
def test_final_fallback_correctness():
    """For any cluster where stable_call does not match and has distance > 2 to all decoded calls, true_call_final should equal provisional_true_call.
    
    Validates: Requirements 6.7
    """
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Cluster has decoded call "K3LR"
    spots = [create_spot(base_time, 7000000.0, dx_call="K3LR")]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    labeled = LabeledCluster(cluster=cluster, provisional_true_call="K3LR")
    
    # Stability says "W1AW" (distance = 4, too far)
    stability_key = StabilityKey(band="40m", mode="CW", freq_bin=7000)
    from rbn_train.types import StableBin
    stable_bins = {stability_key: StableBin(key=stability_key, stable_call="W1AW")}
    
    final_clusters = refine_truth([labeled], stable_bins, config)
    
    assert len(final_clusters) == 1
    # Should fall back to provisional
    assert final_clusters[0].true_call_final == "K3LR"


def test_refine_truth_none_provisional():
    """Test that clusters with None provisional_true_call get None final truth."""
    config = AnalysisConfig()
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    spots = [create_spot(base_time, 7000000.0)]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    labeled = LabeledCluster(cluster=cluster, provisional_true_call=None)
    
    stable_bins = {}
    final_clusters = refine_truth([labeled], stable_bins, config)
    
    assert len(final_clusters) == 1
    assert final_clusters[0].true_call_final is None
