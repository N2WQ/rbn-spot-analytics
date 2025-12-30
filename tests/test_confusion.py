"""Tests for confusion model and priors."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import json
import pytest
from hypothesis import given, strategies as st

from rbn_train.confusion import (
    build_alphabet,
    snr_band_index,
    normalize_call,
    align_calls,
    EditOpKind,
    build_confusion_and_priors,
    export_confusion_model,
)
from rbn_train.priors import Priors, export_priors
from rbn_train.config import AnalysisConfig
from rbn_train.types import Spot, Cluster, ClusterKey, FinalCluster


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


# Feature: rbn-analytics, Property 30: Alphabet construction completeness
@given(charset=st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=65, max_codepoint=90)))
def test_alphabet_construction_completeness(charset):
    """For any charset, the resulting alphabet should contain all characters from charset plus unknown_char if not already present.
    
    Validates: Requirements 7.1
    """
    unknown_char = "?"
    alphabet = build_alphabet(charset, unknown_char)
    
    # All charset characters should be in alphabet
    for char in charset:
        assert char in alphabet.index_by_char
    
    # Unknown char should be in alphabet
    assert unknown_char in alphabet.index_by_char


# Feature: rbn-analytics, Property 31: Character normalization consistency
@given(
    callsign=st.text(min_size=1, max_size=10),
    charset=st.just("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/")
)
def test_character_normalization_consistency(callsign, charset):
    """For any callsign and alphabet, characters not in the alphabet should be replaced with unknown_char.
    
    Validates: Requirements 7.2
    """
    unknown_char = "?"
    alphabet = build_alphabet(charset, unknown_char)
    
    normalized = normalize_call(callsign, alphabet)
    
    # All characters in normalized should be in alphabet
    for char in normalized:
        assert char in alphabet.index_by_char


# Feature: rbn-analytics, Property 32: SNR band assignment correctness
@given(
    snr=st.floats(min_value=-20, max_value=40, allow_nan=False, allow_infinity=False)
)
def test_snr_band_assignment_correctness(snr):
    """For any SNR value and band edges, the assigned band index i should satisfy edges[i] < snr <= edges[i+1].
    
    Validates: Requirements 7.3
    """
    edges = [-999.0, 0.0, 6.0, 12.0, 18.0, 24.0, 999.0]
    
    band_idx = snr_band_index(snr, edges)
    
    # Verify band assignment
    assert 0 <= band_idx < len(edges) - 1
    assert edges[band_idx] < snr <= edges[band_idx + 1]


def test_snr_band_indexing_boundaries():
    """Test SNR band indexing with edges [-999, 0, 6, 12, 18, 24, 999].
    
    Validates: Requirements 7.3
    """
    edges = [-999.0, 0.0, 6.0, 12.0, 18.0, 24.0, 999.0]
    
    # Test boundary values
    assert snr_band_index(-10.0, edges) == 0  # (-999, 0]
    assert snr_band_index(0.0, edges) == 0    # (-999, 0]
    assert snr_band_index(3.0, edges) == 1    # (0, 6]
    assert snr_band_index(6.0, edges) == 1    # (0, 6]
    assert snr_band_index(10.0, edges) == 2   # (6, 12]
    assert snr_band_index(12.0, edges) == 2   # (6, 12]
    assert snr_band_index(15.0, edges) == 3   # (12, 18]
    assert snr_band_index(18.0, edges) == 3   # (12, 18]
    assert snr_band_index(20.0, edges) == 4   # (18, 24]
    assert snr_band_index(24.0, edges) == 4   # (18, 24]
    assert snr_band_index(30.0, edges) == 5   # (24, 999)


def test_align_calls_lz5m_vs_dz5m():
    """Test 'LZ5M' vs 'DZ5M' (single substitution).
    
    Validates: Requirements 7.4
    """
    ops = align_calls("LZ5M", "DZ5M")
    
    # Should have 1 substitution and 3 matches
    subs = [op for op in ops if op.kind == EditOpKind.SUB]
    matches = [op for op in ops if op.kind == EditOpKind.MATCH]
    
    assert len(subs) == 1
    assert subs[0].true_char == "L"
    assert subs[0].obs_char == "D"
    assert len(matches) == 3


def test_align_calls_kc1xx_vs_kc1xnt():
    """Test 'KC1XX' vs 'KC1XNT' (insertion and substitution).
    
    Validates: Requirements 7.4
    """
    ops = align_calls("KC1XX", "KC1XNT")
    
    # Should have edits
    assert len(ops) > 0
    
    # Total edit distance should be 2
    edits = [op for op in ops if op.kind != EditOpKind.MATCH]
    assert len(edits) == 2


def test_align_calls_k3lr_vs_k3l():
    """Test 'K3LR' vs 'K3L' (deletion).
    
    Validates: Requirements 7.4
    """
    ops = align_calls("K3LR", "K3L")
    
    # Should have 1 deletion and 3 matches
    dels = [op for op in ops if op.kind == EditOpKind.DEL]
    matches = [op for op in ops if op.kind == EditOpKind.MATCH]
    
    assert len(dels) == 1
    assert dels[0].true_char == "R"
    assert len(matches) == 3


def test_align_calls_identical():
    """Test identical strings (all matches).
    
    Validates: Requirements 7.4
    """
    ops = align_calls("K3LR", "K3LR")
    
    # Should be all matches
    assert all(op.kind == EditOpKind.MATCH for op in ops)
    assert len(ops) == 4


def test_build_confusion_and_priors_integration():
    """Integration test for building confusion model and priors."""
    config = AnalysisConfig(modes=["CW"], snr_bands=[-999.0, 0.0, 10.0, 999.0])
    base_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Create final clusters
    spots = [
        create_spot(base_time, 7000000.0, mode="CW", dx_call="K3LR", snr_db=5.0),
        create_spot(base_time, 7000000.0, mode="CW", dx_call="K3LR", snr_db=5.0),
    ]
    cluster = Cluster(
        key=ClusterKey(band="40m", mode="CW", freq_bin=14000, time_bin=1000),
        spots=spots
    )
    final_clusters = [FinalCluster(cluster=cluster, true_call_final="K3LR")]
    
    confusion, priors, alphabet = build_confusion_and_priors(final_clusters, config)
    
    # Check priors
    assert "K3LR" in priors.counts
    assert priors.counts["K3LR"] == 1
    
    # Check alphabet
    assert "K" in alphabet.index_by_char
    assert "?" in alphabet.index_by_char


def test_export_confusion_model():
    """Test exporting confusion model to JSON."""
    import numpy as np
    from rbn_train.confusion import ConfusionCounts, Alphabet
    
    alphabet = build_alphabet("ABC", "?")
    modes = ["CW"]
    snr_bands = [-999.0, 0.0, 999.0]
    
    confusion = ConfusionCounts(
        sub_counts=np.zeros((1, 2, 4, 4), dtype=np.int64),
        del_counts=np.zeros((1, 2, 4), dtype=np.int64),
        ins_counts=np.zeros((1, 2, 4), dtype=np.int64),
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        export_confusion_model(confusion, alphabet, modes, snr_bands, temp_path)
        
        # Read back and verify
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        assert data["modes"] == modes
        assert data["snr_band_edges"] == snr_bands
        assert "alphabet" in data
        assert "sub_counts" in data
        
    finally:
        temp_path.unlink()


def test_export_priors():
    """Test exporting priors to JSON."""
    priors = Priors(counts={"K3LR": 10, "W1AW": 5})
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        export_priors(priors, temp_path)
        
        # Read back and verify
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        assert "calls" in data
        assert data["calls"]["K3LR"] == 10
        assert data["calls"]["W1AW"] == 5
        
    finally:
        temp_path.unlink()
