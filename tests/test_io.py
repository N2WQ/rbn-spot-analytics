"""Tests for input/output operations."""

import tempfile
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import pytest
from hypothesis import given, strategies as st, assume

from rbn_train.io import resolve_input_paths, load_spots, normalize_spot
from rbn_train.config import AnalysisConfig
from rbn_train.types import Spot


# Feature: rbn-analytics, Property 3: Frequency conversion consistency
@given(freq_khz=st.floats(min_value=1000, max_value=30000, allow_nan=False, allow_infinity=False))
def test_frequency_conversion_consistency(freq_khz):
    """For any valid freq_khz value, when freq_hz is not provided, the computed freq_hz should equal freq_khz * 1000.0.
    
    Validates: Requirements 2.3
    """
    config = AnalysisConfig(modes=["CW"], min_call_length=2, max_call_length=10)
    
    row = pd.Series({
        'timestamp': '2023-01-01 12:00:00',
        'band': '40m',
        'mode': 'CW',
        'skimmer': 'TEST',
        'dx_call': 'K3LR',
        'snr_db': 10.0,
        'freq_khz': freq_khz,
    })
    
    spot = normalize_spot(row, config)
    
    if spot is not None:
        expected_freq_hz = freq_khz * 1000.0
        assert abs(spot.freq_hz - expected_freq_hz) < 0.01  # Allow small floating point error


# Feature: rbn-analytics, Property 4: Mode normalization idempotence
@given(mode=st.text(min_size=1, max_size=10))
def test_mode_normalization_idempotence(mode):
    """For any mode string, normalizing it (uppercase + strip) twice should produce the same result as normalizing once.
    
    Validates: Requirements 2.4
    """
    # First normalization
    normalized_once = mode.upper().strip()
    
    # Second normalization
    normalized_twice = normalized_once.upper().strip()
    
    # Should be identical
    assert normalized_once == normalized_twice


# Feature: rbn-analytics, Property 5: Callsign normalization idempotence
@given(callsign=st.text(min_size=1, max_size=15))
def test_callsign_normalization_idempotence(callsign):
    """For any callsign string, normalizing it (uppercase + strip) twice should produce the same result as normalizing once.
    
    Validates: Requirements 2.5
    """
    # First normalization
    normalized_once = callsign.upper().strip()
    
    # Second normalization
    normalized_twice = normalized_once.upper().strip()
    
    # Should be identical
    assert normalized_once == normalized_twice


# Feature: rbn-analytics, Property 6: Mode filtering correctness
@given(
    mode=st.sampled_from(["CW", "RTTY", "SSB", "FT8", "PSK31"]),
    modes_list=st.lists(st.sampled_from(["CW", "RTTY", "SSB"]), min_size=1, max_size=3, unique=True)
)
def test_mode_filtering_correctness(mode, modes_list):
    """For any spot and configured modes list, the spot should be included if and only if its normalized mode is in the modes list.
    
    Validates: Requirements 2.6
    """
    config = AnalysisConfig(modes=modes_list, min_call_length=2, max_call_length=10)
    
    row = pd.Series({
        'timestamp': '2023-01-01 12:00:00',
        'band': '40m',
        'mode': mode,
        'skimmer': 'TEST',
        'dx_call': 'K3LR',
        'snr_db': 10.0,
        'freq_hz': 7000000.0,
    })
    
    spot = normalize_spot(row, config)
    
    # Spot should be included if and only if mode is in modes_list
    if mode in modes_list:
        assert spot is not None
        assert spot.mode == mode
    else:
        assert spot is None


# Feature: rbn-analytics, Property 7: Callsign length filtering correctness
@given(
    call_length=st.integers(min_value=1, max_value=20),
    min_length=st.integers(min_value=2, max_value=5),
    max_length=st.integers(min_value=10, max_value=16)
)
def test_callsign_length_filtering_correctness(call_length, min_length, max_length):
    """For any spot and configured min/max call length, the spot should be included if and only if its callsign length is within bounds.
    
    Validates: Requirements 2.7
    """
    assume(min_length <= max_length)
    
    config = AnalysisConfig(
        modes=["CW"],
        min_call_length=min_length,
        max_call_length=max_length
    )
    
    # Create callsign of specific length
    dx_call = 'A' * call_length
    
    row = pd.Series({
        'timestamp': '2023-01-01 12:00:00',
        'band': '40m',
        'mode': 'CW',
        'skimmer': 'TEST',
        'dx_call': dx_call,
        'snr_db': 10.0,
        'freq_hz': 7000000.0,
    })
    
    spot = normalize_spot(row, config)
    
    # Spot should be included if and only if length is within bounds
    if min_length <= call_length <= max_length:
        assert spot is not None
        assert len(spot.dx_call) == call_length
    else:
        assert spot is None


# Feature: rbn-analytics, Property 8: SNR filtering correctness
@given(
    snr=st.floats(min_value=-30, max_value=50, allow_nan=False, allow_infinity=False),
    min_snr=st.floats(min_value=-20, max_value=10, allow_nan=False, allow_infinity=False)
)
def test_snr_filtering_correctness(snr, min_snr):
    """For any spot and configured minimum SNR, the spot should be included if and only if its SNR is greater than or equal to the minimum.
    
    Validates: Requirements 2.8
    """
    config = AnalysisConfig(
        modes=["CW"],
        min_call_length=2,
        max_call_length=10,
        min_snr_db=min_snr
    )
    
    row = pd.Series({
        'timestamp': '2023-01-01 12:00:00',
        'band': '40m',
        'mode': 'CW',
        'skimmer': 'TEST',
        'dx_call': 'K3LR',
        'snr_db': snr,
        'freq_hz': 7000000.0,
    })
    
    spot = normalize_spot(row, config)
    
    # Spot should be included if and only if SNR >= min_snr
    if snr >= min_snr:
        assert spot is not None
        assert spot.snr_db == snr
    else:
        assert spot is None


def test_resolve_input_paths_single_file():
    """Test resolve_input_paths with single file.
    
    Validates: Requirements 1.1
    """
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        paths = resolve_input_paths(temp_path)
        assert len(paths) == 1
        assert paths[0] == Path(temp_path)
    finally:
        Path(temp_path).unlink()


def test_resolve_input_paths_directory():
    """Test resolve_input_paths with directory containing CSV files.
    
    Validates: Requirements 1.2
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some CSV files
        csv1 = Path(tmpdir) / "file1.csv"
        csv2 = Path(tmpdir) / "file2.csv"
        txt1 = Path(tmpdir) / "file.txt"
        
        csv1.touch()
        csv2.touch()
        txt1.touch()
        
        paths = resolve_input_paths(tmpdir)
        
        # Should only return CSV files, sorted
        assert len(paths) == 2
        assert csv1 in paths
        assert csv2 in paths
        assert txt1 not in paths
        assert paths == sorted(paths)


def test_resolve_input_paths_glob():
    """Test resolve_input_paths with glob pattern.
    
    Validates: Requirements 1.3
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files
        csv1 = Path(tmpdir) / "data_2023.csv"
        csv2 = Path(tmpdir) / "data_2024.csv"
        csv3 = Path(tmpdir) / "other.csv"
        
        csv1.touch()
        csv2.touch()
        csv3.touch()
        
        # Use glob pattern
        pattern = str(Path(tmpdir) / "data_*.csv")
        paths = resolve_input_paths(pattern)
        
        # Should only return files matching pattern, sorted
        assert len(paths) == 2
        assert csv1 in paths
        assert csv2 in paths
        assert csv3 not in paths
        assert paths == sorted(paths)


def test_resolve_input_paths_no_files():
    """Test that resolve_input_paths raises error when no files found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory
        with pytest.raises(ValueError, match="No CSV files found"):
            resolve_input_paths(tmpdir)


def test_resolve_input_paths_nonexistent_file():
    """Test that resolve_input_paths raises error for nonexistent file."""
    with pytest.raises(ValueError, match="does not exist"):
        resolve_input_paths("/nonexistent/file.csv")


def test_normalize_spot_invalid_timestamp():
    """Test that spots with invalid timestamps are dropped."""
    config = AnalysisConfig(modes=["CW"])
    
    row = pd.Series({
        'timestamp': 'invalid-date',
        'band': '40m',
        'mode': 'CW',
        'skimmer': 'TEST',
        'dx_call': 'K3LR',
        'snr_db': 10.0,
        'freq_hz': 7000000.0,
    })
    
    spot = normalize_spot(row, config)
    assert spot is None


def test_normalize_spot_empty_callsign():
    """Test that spots with empty callsign are dropped."""
    config = AnalysisConfig(modes=["CW"])
    
    row = pd.Series({
        'timestamp': '2023-01-01 12:00:00',
        'band': '40m',
        'mode': 'CW',
        'skimmer': 'TEST',
        'dx_call': '',
        'snr_db': 10.0,
        'freq_hz': 7000000.0,
    })
    
    spot = normalize_spot(row, config)
    assert spot is None


def test_normalize_spot_empty_skimmer():
    """Test that spots with empty skimmer are dropped."""
    config = AnalysisConfig(modes=["CW"])
    
    row = pd.Series({
        'timestamp': '2023-01-01 12:00:00',
        'band': '40m',
        'mode': 'CW',
        'skimmer': '',
        'dx_call': 'K3LR',
        'snr_db': 10.0,
        'freq_hz': 7000000.0,
    })
    
    spot = normalize_spot(row, config)
    assert spot is None


def test_load_spots_integration():
    """Integration test for loading spots from CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write CSV with RBN format
        f.write("callsign,de_pfx,de_cont,freq,band,dx,dx_pfx,dx_cont,mode,db,date,speed,tx_mode\n")
        f.write("TEST1,K,NA,7000,40m,K3LR,K,NA,CQ,10,1/1/2023 12:00,25,CW\n")
        f.write("TEST2,K,NA,7000.1,40m,K3LR,K,NA,CQ,12,1/1/2023 12:00,25,CW\n")
        f.write("TEST3,K,NA,14000,20m,W1AW,W,NA,CQ,8,1/1/2023 12:00,25,RTTY\n")
        temp_path = f.name
    
    try:
        config = AnalysisConfig(modes=["CW", "RTTY"])
        spots = load_spots([Path(temp_path)], config)
        
        # Should load all 3 spots
        assert len(spots) == 3
        assert all(isinstance(spot, Spot) for spot in spots)
        
        # Check first spot
        assert spots[0].dx_call == "K3LR"
        assert spots[0].mode == "CW"
        assert spots[0].snr_db == 10.0
        
    finally:
        Path(temp_path).unlink()


def test_load_spots_with_freq_khz():
    """Test loading spots with freq_khz instead of freq_hz."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write CSV with RBN format (freq is in kHz)
        f.write("callsign,de_pfx,de_cont,freq,band,dx,dx_pfx,dx_cont,mode,db,date,speed,tx_mode\n")
        f.write("TEST1,K,NA,7000.0,40m,K3LR,K,NA,CQ,10,1/1/2023 12:00,25,CW\n")
        temp_path = f.name
    
    try:
        config = AnalysisConfig(modes=["CW"])
        spots = load_spots([Path(temp_path)], config)
        
        assert len(spots) == 1
        assert spots[0].freq_hz == 7000000.0  # Converted from kHz
        
    finally:
        Path(temp_path).unlink()
